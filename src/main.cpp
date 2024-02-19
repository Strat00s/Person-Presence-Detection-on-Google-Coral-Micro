//TODO move drawing into web page?

//TODO output to pin
//TODO output on LED
//TODO output to uri

//TODO mask threshold wording

//TODO configurable grid size
//TODO rotation


#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <deque>
#include <cstdarg>
#include <cstdlib>
#include <cerrno>

#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "libs/base/filesystem.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "libs/base/gpio.h"
#include "libs/tpu/edgetpu_op.h"

#include "libs/base/ethernet.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/dns.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/tcpip.h"
#include "libs/base/check.h"
//#include "x86_64-linux-gnu/sys/time.h"

#include "local_libs/http_server.h"
#include "structs.hpp"
#include "ipc_message.hpp"
#include "AwaitQueue.hpp"

using namespace coralmicro;
using namespace std;


#define MAIN_WEB_PATH             "/page.html"
#define STREAM_WEB_PATH           "/camera_stream"
#define DETECTION_STATUS_WEB_PATH "/detection_status"
#define CONFIG_SAVE_WEB_PATH      "/config_save"
#define CONFIG_LOAD_WEB_PATH      "/config_load"
#define CONFIG_RESET_WEB_PATH     "/config_reset"


#define MODEL_PATH            "/model.tflite"
#define CONFIG_PATH           "/cfg.csv"

#define DEFAULT_IMAGE_SIZE   320
#define DEFAULT_MASK_SIZE    32
#define DEFAULT_ROTATION     270
#define DEFAULT_DET_THRESH   50
#define DEFAULT_IOU_THRESH   50
#define DEFAULT_FP_CHANGE    40
#define DEFAULT_FP_COUNT     5
#define DEFAULT_JPEG_QUALITY 75
#define DEFAULT_MASK_THRESH  50

#define TENSOR_ARENA_SIZE     8 * 1024 * 1024
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);


//int image_size   = DEFAULT_IMAGE_SIZE;
tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter; //must be created and configured in the main task for some reason
tflite::MicroErrorReporter error_reporter;

int image_size = DEFAULT_IMAGE_SIZE;
cfg_struct_t global_config;
bool update_config = true;
SemaphoreHandle_t config_mux;

TaskHandle_t detect_task_handle; // detector task handle

AwaitQueue<web_task_msg_t> result_queue(2);


void exit() {
    printf("Exiting...\n");
    vTaskSuspend(NULL);
}

/*----(CONFIG HANDLING)----*/
cfg_struct_t buildDefaultConfig() {
    cfg_struct_t cfg;
    cfg.mask_size    = DEFAULT_MASK_SIZE;
    cfg.mask_thresh  = DEFAULT_MASK_THRESH;
    cfg.rotation     = DEFAULT_ROTATION;
    cfg.det_thresh   = DEFAULT_DET_THRESH;
    cfg.iou_thresh   = DEFAULT_IOU_THRESH;
    cfg.fp_change    = DEFAULT_FP_CHANGE;
    cfg.fp_count     = DEFAULT_FP_COUNT;
    cfg.jpeg_quality = DEFAULT_JPEG_QUALITY;
    cfg.mask.resize(cfg.mask_size * cfg.mask_size, 0);
    return cfg;
}

/** @brief Generate config from csv
 * 
 * @param csv 
 * @param config 
 * @return true on success, false on invalid csv (csv has less tha 8 fields)
 */
bool configFromCsv(const std::string &csv, cfg_struct_t *config) {
    if (std::count(csv.begin(), csv.end(), ',') < 8) {
        printf("invalid csv\n");
        return false;
    }

    bool malformed = false;
    bool finished = false;
    int field_num = -1;
    size_t start = 0;
    size_t end = csv.find(',');
    cfg_struct_t new_config = buildDefaultConfig();

    while (!finished) {
        field_num++;
        if (end == std::string::npos) {
            finished = true;
            end = csv.size();
        }
        std::string field_val = csv.substr(start, end - start);

        printf("%d %s %d %d\n", field_num, field_val.c_str(), start, end);
        vTaskDelay(10);

        start = end + 1;
        end = csv.find(',', start);

        //skip malformed fields
        if (field_val.size() == 0) {
            malformed = true;
            continue;
        }

        //mask field
        if (field_num == 2) {
            new_config.mask.clear();
            new_config.mask.reserve(field_val.size());
            for (size_t i = 0; i < field_val.size(); i++)
                new_config.mask.push_back(field_val[i] == '1');
            continue;
        }

        //try to convert the field to number
        int val = 0;
        char *endptr;
        val = std::strtol(field_val.c_str(), &endptr, 10);
        if (*endptr) {
            printf("Failed to convert %s to int\n", field_val.c_str());
            malformed = true;
            continue;
        }

        //everything but mask
        switch (field_num) {
            case 0: new_config.mask_size    = val; break;
            case 1: new_config.mask_thresh  = val; break;
            case 3: new_config.rotation     = val; break;
            case 4: new_config.det_thresh   = val; break;
            case 5: new_config.iou_thresh   = val; break;
            case 6: new_config.fp_change    = val; break;
            case 7: new_config.fp_count     = val; break;
            case 8: new_config.jpeg_quality = val; break;
            default: printf("Unhandled field %d: %s\n", field_num, field_val.c_str()); break;
        }
    }

    if (malformed)
        new_config.mask.resize(new_config.mask_size * new_config.mask_size, 0);

    *config = new_config;
    return true;
}

/** @brief 
 * 
 * @param config 
 * @return std::string 
 */
std::string csvFromConfig(const cfg_struct_t &config) {
    std::string csv = std::to_string(config.mask_size) + ',';
    csv += std::to_string(config.mask_thresh) + ',';
    for (auto val: config.mask)
        csv += std::to_string(val);
    csv += ',';
    csv += std::to_string(config.rotation)   + ',';
    csv += std::to_string(config.det_thresh) + ',';
    csv += std::to_string(config.iou_thresh) + ',';
    csv += std::to_string(config.fp_change)  + ',';
    csv += std::to_string(config.fp_count)   + ',';
    csv += std::to_string(config.jpeg_quality);
    return csv;
}

/** @brief Write csv config to file
 * 
 * @param csv 
 * @return true on success, false otherwise
 */
bool writeConfigFile(const std::string &csv) {
    printf("Write config: %s\n", csv.c_str());
    return LfsWriteFile(CONFIG_PATH, csv);
}

/** @brief Read csv config from file
 * 
 * @return empty string on failure
 */
std::string readConfigFile() {
    std::string config_csv;
    if (!LfsReadFile(CONFIG_PATH, &config_csv))
        return {};
    return config_csv;
}


/*----(IMAGE DRAWING)----*/
/** @brief Simple function to draw rectangles into raw picture
 * 
 * @param bbox bounding box to be drawn
 * @param image image to which to draw
 * @param image_size image size
 * @param r red channel value
 * @param g green channel value
 * @param b bluc channel value
 */
void drawRectangleFast(bbox_t bbox, image_vector_t *image, int image_size, uint8_t r, uint8_t g, uint8_t b) {
    if (bbox.xmin < 0)
        bbox.xmin = 0;
    if (bbox.ymin < 0)
        bbox.ymin = 0;
    if (bbox.xmax > image_size - 1)
        bbox.xmax = image_size - 1;
    if (bbox.ymax > image_size - 1)
        bbox.ymax = image_size - 1;

    //draw top and bottom sides
    int max_start = (bbox.ymax * image_size + bbox.xmin) * 3;
    int min_start = (bbox.ymin * image_size + bbox.xmin) * 3;
    int step = 3;
    int end = (bbox.xmax - bbox.xmin) * step;
    for (int i = 0; i < end; i += step) {
        (*image)[max_start + i]     = r;
        (*image)[max_start + i + 1] = g;
        (*image)[max_start + i + 2] = b;
        (*image)[min_start + i]     = r;
        (*image)[min_start + i + 1] = g;
        (*image)[min_start + i + 2] = b;
    }

    //draw left and right sides
    max_start = (bbox.ymin * image_size + bbox.xmax) * 3;
    step = image_size * 3;
    end = (bbox.ymax - bbox.ymin) * step;
    for (int i = 0; i < end; i += step) {
        (*image)[max_start + i]     = r;
        (*image)[max_start + i + 1] = g;
        (*image)[max_start + i + 2] = b;
        (*image)[min_start + i]     = r;
        (*image)[min_start + i + 1] = g;
        (*image)[min_start + i + 2] = b;
    }
}


/*----(MASK HANDLING)----*/
// Helper function to check if the coordinates are inside the box.
inline bool isInsideBox(int x, int y, int x_min, int x_max, int y_min, int y_max) {
    return (x >= x_min && x <= x_max && y >= y_min && y <= y_max);
}

/** @brief Check if enough bbox is masked
 * 
 * @param bbox 
 * @param mask 
 * @param mask_size 
 * @param image_size 
 * @param threshold
 * @return true if masked, false if not
 */
bool isBBoxMasked(bbox_t bbox, const image_vector_t& mask, int mask_size, int image_size, float threshold) {
    // Scale factor from the image size to grid size.
    const int scale = image_size / mask_size;

    int covered_cells = 0;
    int total_cells = 0;

    // Iterate over the box's range.
    for (int y = bbox.ymin; y <= bbox.ymax; y++) {
        for (int x = bbox.xmin; x <= bbox.xmax; x++) {
            // Map the image coordinates to grid coordinates.
            int gridX = x / scale;
            int gridY = y / scale;

            // Check if the current point is inside the box.
            if (isInsideBox(gridX, gridY, bbox.xmin/scale, bbox.xmax/scale, bbox.ymin/scale, bbox.ymax/scale)) {
                // Index in the grid
                int index = gridY * mask_size + gridX;

                // Check if the cell is non-toggled
                covered_cells += mask[index];
                total_cells++;
            }
        }
    }

    return covered_cells > total_cells * (1.0f - threshold);
}


/*----(DETECTION HADNLING)----*/
/** @brief Function to check if the specified region has changed significantly - average color approach
 * 
 * @param current_image 
 * @param previous_image 
 * @param image_size 
 * @param bbox bounding vox
 * @param change How much individual pixels need to change to be counted
 * @param count How many pixels have to change
 * @return true if changed enough, false otherwise
 */
bool hasBBoxChanged(const image_vector_t &current_image, const image_vector_t &previous_image, int image_size, bbox_t bbox, float change, float count) {
    int32_t changed = 0;
    int32_t pixel_cnt = (bbox.xmax - bbox.xmin + 1) * (bbox.ymax - bbox.ymin + 1);
    for (int y = bbox.ymin; y <= bbox.ymax; ++y) {
        for (int x = bbox.xmin; x <= bbox.xmax; ++x) {
            size_t index = (y * image_size + x) * 3;
            int pixel_change = std::abs(current_image[index] - previous_image[index]) +
                               std::abs(current_image[index + 1] - previous_image[index + 1]) +
                               std::abs(current_image[index + 2] - previous_image[index + 2]);
            if (pixel_change > 0xFF * 3 * change)
                changed++;
        }
    }

    return changed > std::round(pixel_cnt * count);
}

/** @brief Intersection over union calculations for overlaping bounding boxes
 * 
 * @param a First bounding box
 * @param b Second boundig box
 * @return overlap ratio
 */
float IoU(const bbox_t& a, const bbox_t& b) {
    float intersectionArea = std::max(0, std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin)) *
                             std::max(0, std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin));
    float unionArea = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                      (b.xmax - b.xmin) * (b.ymax - b.ymin) - intersectionArea;
    return intersectionArea / unionArea;
}

/** @brief Get results from inference.
 * 1. get all bboxes with person id
 * 2. NMS
 * 3. calculate masked and detected
 * 3. calculate change against background
 * 
 * @param results vector where to store resulting bboxes
 * @param interpreter tflite interpreter
 * @param image current image
 * @param background last background image to compare against
 * @param config current config
 * @return 0 if nothing is detected.
 * 1 if something is detected and masked.
 * 2 if something is/was detected and is/was unmasked in last 5 frames.
 * 3 if something is detected, is unmasked and is detected in the last 3 out of 5 frames.
 */
int getResults(std::vector<bbox_t> *results, tflite::MicroInterpreter *interpreter, const image_vector_t &image,
                                                    const image_vector_t &background, const cfg_struct_t &config) {

    static int consec_unmasked = 0; //keeps count of consecutive unmasked detections

    float det_thresh  = config.det_thresh / 100.0f;
    float iou_thresh  = config.iou_thresh / 100.0f;
    float fp_change   = config.fp_change  / 100.0f;
    float fp_count    = config.fp_count   / 100.0f;
    float mask_thresh = config.mask_thresh / 100.0f;
    //get data from output tensor
    float *bboxes, *ids, *scores, *count;
    if (interpreter->output_tensor(2)->dims->size == 1) {
        scores = tflite::GetTensorData<float>(interpreter->output_tensor(0));
        bboxes = tflite::GetTensorData<float>(interpreter->output_tensor(1));
        count  = tflite::GetTensorData<float>(interpreter->output_tensor(2));
        ids    = tflite::GetTensorData<float>(interpreter->output_tensor(3));
    }
    else {
        bboxes = tflite::GetTensorData<float>(interpreter->output_tensor(0));
        ids    = tflite::GetTensorData<float>(interpreter->output_tensor(1));
        scores = tflite::GetTensorData<float>(interpreter->output_tensor(2));
        count  = tflite::GetTensorData<float>(interpreter->output_tensor(3));
    }

    bbox_vector_t tmp_results;
    results->clear();
    size_t cnt = static_cast<int>(count[0]);
    for (int i = 0; i < cnt; ++i) {
        //skip object with id != 0 (not person) and below threshold
        if (std::round(ids[i]) || scores[i] < det_thresh)
            continue;

        bbox_t bbox = {
            .xmin  = std::max(0.0f, std::round(bboxes[4 * i + 1] * image_size)),
            .ymin  = std::max(0.0f, std::round(bboxes[4 * i]     * image_size)),
            .xmax  = std::max(0.0f, std::round(bboxes[4 * i + 3] * image_size)),
            .ymax  = std::max(0.0f, std::round(bboxes[4 * i + 2] * image_size)),
            .score = std::round(scores[i] * 100),
            .type  = 0
        };
        if (bbox.xmax >= image_size)
            bbox.xmax = image_size - 1;
        if (bbox.ymax >= image_size)
            bbox.ymax = image_size - 1;
        tmp_results.push_back(bbox);
    }

    //nothing valid detected
    if (!tmp_results.size()) {
        results->clear();
        if (consec_unmasked > 0)
            consec_unmasked--;
        return consec_unmasked > 0 ? 2 : 0;
    }

    //single object detected
    if (tmp_results.size() == 1) {
        //change type if unmasked
        if (!isBBoxMasked(tmp_results[0], config.mask, config.mask_size, image_size, mask_thresh)) {
            if (consec_unmasked < 5)
                consec_unmasked++;
            tmp_results[0].type = consec_unmasked < 3 ? 1 : 2;
            *results = tmp_results;
            return consec_unmasked < 3 ? 2 : 3;
        }
        //decrease if masked
        if (consec_unmasked > 0)
            consec_unmasked--;
        *results = tmp_results;
        return consec_unmasked > 0 ? 2 : 1;
    }

    // order indices to the tmp_results from highest to lowest scores
    std::vector<int> indices(tmp_results.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return tmp_results[lhs].score > tmp_results[rhs].score;
    });

    bool once = false;
    while (!indices.empty()) {
        int idx = indices.front();
        indices.erase(indices.begin());
        auto current_box = tmp_results[idx];

        // NMS
        // probably not required as the model has it inside
        // go through remaining box and check if there are any that overlap with the current box and have lower score
        for (auto it = indices.begin(); it != indices.end(); ) {
            //remove index to box that is overlapped or move to next one if not overlapped
            if (IoU(current_box, tmp_results[*it]) > iou_thresh)
                it = indices.erase(it);
            else
                it++;
        }

        //TODO join haschange and ismasked to single function (single loop)
        //ignore boxes with not enough change
        if (!hasBBoxChanged(image, background, image_size, current_box, fp_change, fp_count)) {
            //printf("False positive %d\n");
            continue;
        }

        //check if box is not masked and update consecutive detection counter and pick right type
        if (!isBBoxMasked(current_box, config.mask, config.mask_size, image_size, mask_thresh)) {
            //unmasked box -> increase conscutive detection (but only once, since it is per frame)
            if (!once) {
                once = true;
                if (consec_unmasked < 5)
                    consec_unmasked++;
            }
            current_box.type = consec_unmasked <= 3 ? 1 : 2;
        }
        results->push_back(current_box);
    }

    //all boxes were masked -> decrease
    if (!once && consec_unmasked > 0)
        consec_unmasked--;

    //something is detected, but it is masked for too many frames
    if (!consec_unmasked)
        return 1;
    else
        return consec_unmasked > 0 ? 2 : 0;
        //return consec_unmasked <= 3 ? 2 : 3;
}


//Main detect task
static void detectTask(void *args) {
    int ret;
    int status = 0;
    int clean = 0;
    int bckg_upd_cnt = 0;

    cfg_struct_t config = global_config;
    image_vector_t current_image(image_size * image_size * 3);
    image_vector_t background(image_size * image_size * 3);
    bbox_vector_t results;

    CameraFrameFormat frame = {
        CameraFormat::kRgb,
        CameraFilterMethod::kBilinear,
        CameraRotation::k270,
        image_size,
        image_size,
        true,
        current_image.data(),
        true
    };

    TickType_t last_wake_time;
    const TickType_t frequency = 50;
    while (true) {
        vTaskDelayUntil(&last_wake_time, frequency);

        //get image
        CameraTask::GetSingleton()->DiscardFrames(1);
        ret = CameraTask::GetSingleton()->GetFrame({frame});
        if (!ret) {
            printf("Failed to get image from camera.\r\n");
            status = -1;
            continue;
        }

        //slow because data copying
        std::memcpy(tflite::GetTensorData<uint8_t>(interpreter->input_tensor(0)), current_image.data(), current_image.size());

        // run model
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            status = -2;
            continue;
        }

        if (update_config) {
            xSemaphoreTake(config_mux, portMAX_DELAY);
            config = global_config;
            update_config = false;
            xSemaphoreGive(config_mux);
        }

        int det_status = getResults(&results, interpreter, current_image, background, config);
        bckg_upd_cnt = det_status ? 0 : bckg_upd_cnt + 1;

        LedSet(Led::kUser, det_status > 1);
        printf("Detection status: %d\n", det_status);

        if (bckg_upd_cnt >= 10) {
            background = current_image;
            printf("Background updated\n");
        }

        //slow because data copying
        result_queue.push({det_status, current_image, results}, portMAX_DELAY, true);
    }
}


/*----(WEB HADNLING)----*/
HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "index.html"))
        return std::string(MAIN_WEB_PATH);
    
    else if (StrEndsWith(uri, STREAM_WEB_PATH)) {
        //TODO wait for inference to be done
        web_task_msg_t results;
        int ret = result_queue.pop(&results, 100, 100);
        if (ret) {
            printf("Failed to pop item from queue\n");
            return {};
        }

        int ticks = xTaskGetTickCount();
        for (size_t i = 0; i < results.bboxes.size(); i++) {
            switch (results.bboxes[i].type) {
                case 0: drawRectangleFast(results.bboxes[i], &results.image, image_size, 0,     0, 255); break;
                case 1: drawRectangleFast(results.bboxes[i], &results.image, image_size, 255, 255,   0); break;
                case 2: drawRectangleFast(results.bboxes[i], &results.image, image_size, 0,   255,   0); break;
            }
        }
        printf("Draw time: %dms\n", xTaskGetTickCount() - ticks);
        //not raw jpeg. Last byte is detection status
        image_vector_t jpeg;
        JpegCompressRgb(results.image.data(), image_size, image_size, global_config.jpeg_quality, &jpeg);
        jpeg.push_back(results.status);
        return jpeg;
    }

    else if (StrEndsWith(uri, DETECTION_STATUS_WEB_PATH)) {
        web_task_msg_t results;
        int ret = result_queue.pop(&results, 100, 100);
        if (ret) {
            printf("Failed to pop item from queue\n");
            return {};
        }
        std::vector<uint8_t> status;
        status.push_back(results.status);
        return status;
    }

    else if (StrEndsWith(uri, CONFIG_LOAD_WEB_PATH)) {
        xSemaphoreTake(config_mux, portMAX_DELAY);
        std::string csv = csvFromConfig(global_config);
        xSemaphoreGive(config_mux);
        return std::vector<uint8_t>(csv.begin(), csv.end());
    }

    else if (StrEndsWith(uri, CONFIG_RESET_WEB_PATH)) {
        xSemaphoreTake(config_mux, portMAX_DELAY);
        global_config = buildDefaultConfig();
        std::string csv = csvFromConfig(global_config);
        update_config = true;
        xSemaphoreGive(config_mux);
        if (!writeConfigFile(csv)) {
            printf("Failed to write reset config to file\n");
        }
        return std::vector<uint8_t>(csv.begin(), csv.end());
    }

    return {};
}

std::string postUriHandler(std::string uri, std::vector<uint8_t> payload) {
    printf("on post finished\r\n");

    if (StrEndsWith(uri, CONFIG_SAVE_WEB_PATH)) {
        std::string csv = std::string(payload.begin(), payload.end());

        xSemaphoreTake(config_mux, portMAX_DELAY);
        if (!configFromCsv(csv, &global_config)) {
            printf("Invalid CSV provided\n");
        }
        update_config = true;
        xSemaphoreGive(config_mux);

        if (!writeConfigFile(csv)) {
            printf("Failed to write csv to file\n");
        }

        return "/200.html";
    }

    printf("unhandled uri: %s\n", uri.c_str());
    return {};
}



extern "C" [[noreturn]] void app_main(void* param) {
    (void)param;

    config_mux = xSemaphoreCreateMutex();

    LedSet(Led::kStatus, true);

    vTaskDelay(5000);
    
    coralmicro::

    //load config
    std::string csv = readConfigFile();
    printf("Read csv: %s\n", csv.c_str());
    //empty/non-existant file or malformed csv
    if (!csv.size() || !configFromCsv(csv, &global_config)) {
        global_config = buildDefaultConfig();
        if (writeConfigFile(csvFromConfig(global_config))) {
            printf("Failed to write config file\n");
            vTaskDelete(NULL);
        }
        printf("Config regenerated\n");
    }
    else
        printf("Config loaded from file\n");


    // Try and get an IP
    printf("Attempting to use ethernet...\r\n");
    if (!EthernetInit(true)) {
        //TODO run without ethernet?
        printf("Failed to init ethernet\n");
        exit();
    }
    auto ip_addr = EthernetGetIp();
    if (ip_addr.has_value())
        printf("DHCP succeeded, our IP is %s\n", ip_addr.value().c_str());
    else {
        printf("We didn't get an IP via DHCP, not progressing further.\r\n");
        exit();
    }



    //Read the model and checks version.
    std::vector<uint8_t> model_raw;  //entire model is stored in this vector
    if (!LfsReadFile(MODEL_PATH, &model_raw)) {
        TF_LITE_REPORT_ERROR(&error_reporter, "Failed to load model!");
        vTaskSuspend(nullptr);
    }

    //confirm model scheme version
    auto* model = tflite::GetModel(model_raw.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(&error_reporter, "Model schema version is %d, supported is %d", model->version(), TFLITE_SCHEMA_VERSION);
        vTaskSuspend(nullptr);
    }

    //create micro interpreter
    resolver.AddDequantize();
    resolver.AddDetectionPostprocess();
    resolver.AddCustom(kCustomOp, RegisterCustomOp());
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &error_reporter);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&error_reporter, "AllocateTensors failed.");
        vTaskSuspend(nullptr);
    }

    if (interpreter->inputs().size() != 1) {
        printf("ERROR: Model must have only one input tensor\n");
        exit();
    }

    auto input_tensor = interpreter->input_tensor(0);
    if (input_tensor->dims->data[1] != input_tensor->dims->data[2]) {
        printf("Model input is not square\n");
        exit();
    }

    if (interpreter->outputs().size() != 4) {
        printf("Output size mismatch\\n");
        exit();
    }

    //Turn on the TPU and get it's context.
    auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
    if (!tpu_context) {
        TF_LITE_REPORT_ERROR(&error_reporter, "ERROR: Failed to get EdgeTpu context!");
        vTaskSuspend(nullptr);
    }

    //start camera
    CameraTask::GetSingleton()->SetPower(false);
    vTaskDelay(100);
    CameraTask::GetSingleton()->SetPower(true);
    CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

    //start detection task
    xTaskCreate(detectTask, "detect", 16000, nullptr, 4, &detect_task_handle);

    //start http server
    PostHttpServer http_server;
    http_server.registerPostUriHandler(postUriHandler);
    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);

    vTaskSuspend(nullptr);
}