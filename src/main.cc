//TODO mark detected after 3 out of 5 consecutive frames

//TODO final structure:
//bbox
//score
//type of detection (false positive, masked, unmasked but not yet full, full)

//TODO move stuff to m4 core
//TODO move stuff to own classes

//TODO output to pin
//TODO output on LED
//TODO output to uri

//TODO remove useless data copying (if reference can be used)

//TODO fix isCovered

//TODO has changed average -> better calculcations

//TODO detect brightness change
//if big enough, update current last frame even if something is detected (update only parts with not detections)

//TODO configurable grid size
//TODO rotation
//TODO status update to webpage

//TODO UI???

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <deque>

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

using namespace coralmicro;
using namespace std;


#define MAIN_WEB_PATH         "/page.html"
#define STREAM_WEB_PATH       "/camera_stream"
#define CONFIG_SAVE_WEB_PATH  "/config_save"
#define CONFIG_LOAD_WEB_PATH  "/config_load"
#define CONFIG_RESET_WEB_PATH "/config_reset"
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


#define IDX(x, y, width) (y * width + x) * 3;

#define TENSOR_ARENA_SIZE     8 * 1024 * 1024
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);


typedef std::vector<uint8_t> imageVector_t;


int image_size   = DEFAULT_IMAGE_SIZE;
typedef struct cfg_struct{
    int mask_size    = DEFAULT_MASK_SIZE;
    std::vector<uint8_t> mask;
    int rotation     = DEFAULT_ROTATION;
    int det_thresh   = DEFAULT_DET_THRESH;
    int iou_thresh   = DEFAULT_IOU_THRESH;
    int fp_change    = DEFAULT_FP_CHANGE;
    int fp_count     = DEFAULT_FP_COUNT;
    int jpeg_quality = DEFAULT_JPEG_QUALITY;
    cfg_struct() {mask.resize(mask_size * mask_size, 0);}
} cfg_struct_t;
cfg_struct_t config;


tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter; //must be created and configured in the main task for some reason
tflite::MicroErrorReporter error_reporter;

imageVector_t image;
CameraFrameFormat frame;

SemaphoreHandle_t status_mux;
SemaphoreHandle_t image_mux;
SemaphoreHandle_t result_mux;
SemaphoreHandle_t config_mux;
int status;

TaskHandle_t detect_task_handle; // detector task handle

imageVector_t last_frame; // last empty frame before any detection happens
int detected = 0;


typedef struct {
    int ymin;
    int xmin;
    int ymax;
    int xmax;
    int score;
} bbox_t;
std::vector<bbox_t> results;


int detect_cnt = 0;
std::deque<bool> detected_buf(5, false);


/*----(CONFIG HANDLING)----*/
/** @brief build config csv from current config 
 * 
 * @param config 
 * @return std::string 
 */
std::string buildConfigCsv(cfg_struct_t config) {
    std::string cfg = std::to_string(config.mask_size) + ',';
    for (auto val: config.mask)
        cfg += std::to_string(val);
    cfg += ',';
    cfg += std::to_string(config.rotation)   + ',';
    cfg += std::to_string(config.det_thresh) + ',';
    cfg += std::to_string(config.iou_thresh) + ',';
    cfg += std::to_string(config.fp_change)  + ',';
    cfg += std::to_string(config.fp_count)   + ',';
    cfg += std::to_string(config.jpeg_quality);
    return cfg;
}

/** @brief Partse config csv string and save it to global variables
 * 
 * @param config_csv string with valid csv
 * @return 0 on success, 1 on failure to acquire mutex 
*/
int saveConfig(std::string config_csv) {
    if (std::count(config_csv.begin(), config_csv.end(), ',') < 7) {
        printf("invalid csv\n");
        return 2;
    }

    bool malformed = false;
    bool finished = false;
    int field_num = -1;
    size_t start = 0;
    size_t end = config_csv.find(',');
    cfg_struct_t new_config;

    if (!new_config.mask.size()) {
        printf("UH OH\n");
        vTaskDelay(100000);
    }

    while (!finished) {
        field_num++;
        if (end == std::string::npos) {
            finished = true;
            end = config_csv.size();
        }
        std::string field_val = config_csv.substr(start, end - start);
        printf("%d %s %d %d\n", field_num, field_val.c_str(), start, end);

        vTaskDelay(10);

        start = end + 1;
        end = config_csv.find(',', start);

        if (field_val.size() == 0) {
            malformed = true;
            continue;
        }

        if (field_num == 1) {
            new_config.mask.clear();
            new_config.mask.reserve(field_val.size());
            for (size_t i = 0; i < field_val.size(); i++)
                new_config.mask.push_back(field_val[i] == '1');
            continue;
        }

        int val = 0;
        try {
            val = std::stoi(field_val);
        }
        catch(const std::exception& e) {
            printf("%s\n", e.what());
            continue;
        }

        switch (field_num) {
            case 0: new_config.mask_size    = val; break;
            case 2: new_config.rotation     = val; break;
            case 3: new_config.det_thresh   = val; break;
            case 4: new_config.iou_thresh   = val; break;
            case 5: new_config.fp_change    = val; break;
            case 6: new_config.fp_count     = val; break;
            case 7: new_config.jpeg_quality = val; break;
            default: printf("Unhandled field %d: %s\n", field_num, field_val.c_str()); break;
        }
    }

    if (malformed) {
        new_config.mask.resize(new_config.mask_size * new_config.mask_size, 0);
        LfsWriteFile(CONFIG_PATH, buildConfigCsv(new_config));
    }

    if (xSemaphoreTake(config_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
        config = new_config;
        xSemaphoreGive(config_mux);
        printf("config saved\n");
        return 0;
    }

    printf("Failed to get config mutex\n");
    return 1;
}

/** @brief Save configuration csv (in string form) to file and update global variables
 * 
 * @param config_csv csv string
 * @return int 0 on success, 1 on failure to write to file, 2 when failed to update global variables
 */
int saveConfigFile(std::string config_csv) {
    //write the csv string to file
    if (!LfsWriteFile(CONFIG_PATH, config_csv))
        return 3;

    //save it to config variable
    return saveConfig(config_csv);
}

/** @brief Load config csv from a file or generate it if file does not exist and update global variables
 * 
 * @return config csv string on success, empty on failure
 */
int loadConfigFile(std::string *str) {
    if (!LfsFileExists(CONFIG_PATH)) {
        *str = buildConfigCsv(config);
        saveConfigFile(*str);
        return 0;
    }

    if (!LfsReadFile(CONFIG_PATH, str))
        return 1;

    if (saveConfig(*str))
        return -2;

    return 0;
}


/*----(IMAGE DRAWING)----*/
void drawRectangleFast(int xmin, int ymin, int xmax, int ymax, imageVector_t *image, int image_size, uint8_t r, uint8_t g, uint8_t b) {
    if (xmin < 0)
        xmin = 0;
    if (ymin < 0)
        ymin = 0;
    if (xmax > image_size)
        xmax = image_size;
    if (ymax > image_size)
        ymax = image_size;

    //draw top and bottom sides
    int max_start = (ymax * image_size + xmin) * 3;
    int min_start = (ymin * image_size + xmin) * 3;
    int step = 3;
    for (int i = 0; i < (xmax - xmin) * step; i += step) {
        (*image)[min_start + i]     = r;
        (*image)[min_start + i + 1] = g;
        (*image)[min_start + i + 2] = b;
        (*image)[max_start + i]     = r;
        (*image)[max_start + i + 1] = g;
        (*image)[max_start + i + 2] = b;
    }

    //draw left and right sides
    max_start = (ymin * image_size + xmax) * 3;
    step = image_size * 3;
    for (int i = 0; i < (ymax - ymin) * step; i += step) {
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

bool isMasked(int x_min, int x_max, int y_min, int y_max, int image_size, const imageVector_t& mask, int mask_size) {
    // Scale factor from the image size to grid size.
    const int scale = image_size / mask_size;

    int coveredCells = 0;
    int totalCells = 0;

    // Iterate over the box's range.
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            // Map the image coordinates to grid coordinates.
            int gridX = x / scale;
            int gridY = y / scale;

            // Check if the current point is inside the box.
            if (isInsideBox(gridX, gridY, x_min/scale, x_max/scale, y_min/scale, y_max/scale)) {
                // Index in the grid
                int index = gridY * mask_size + gridX;

                // Check if the cell is non-toggled
                coveredCells += mask[index];
                totalCells++;
            }
        }
    }

    //printf("%d %d %d %d\n", coveredCells, totalCells, mask_size, mask.size());

    // Check if at least half of the box is covered by non-toggled cells.
    return coveredCells > totalCells / 2;
}


//TODO move into own class
void setStatus(int s) {
    if (xSemaphoreTake(status_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
        status = s;
        xSemaphoreGive(status_mux);
    }
    else
        printf("failed to acquire det mutex\n");
}

// Function to check if the specified region has changed significantly - average color approach
bool hasChangedAverage(const imageVector_t &currentImage, const imageVector_t &previousImage, int image_size, int xmin, int ymin, int xmax, int ymax, float change, float count) {
    int32_t changed = 0;
    int32_t pixel_cnt = (xmax - xmin + 1) * (ymax - ymin + 1);
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            size_t index = (y * image_size + x) * 3;
            int pixel_change = std::abs(currentImage[index] - previousImage[index]) +
                               std::abs(currentImage[index + 1] - previousImage[index + 1]) +
                               std::abs(currentImage[index + 2] - previousImage[index + 2]);
            if (pixel_change > 255 * change)
                changed++;
        }
    }

    //printf("changed, pixelcnt: %ld %ld\n", changed, pixel_cnt);

    return changed > (int32_t)std::round(pixel_cnt * count);
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

//std::vector<tensorflow::Object> getResults(tflite::MicroInterpreter *interpreter, float threshold, float iou_threshold) {
std::vector<bbox_t> getResults(tflite::MicroInterpreter *interpreter, const imageVector_t &image, const imageVector_t &last_image,
                               int image_size, float threshold, float iou_threshold, float fp_change, float fp_count) {
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

    detected = 0;
    size_t cnt = static_cast<size_t>(count[0]);
    std::vector<bbox_t> tmp_results, results;
    for (size_t i = 0; i < cnt; ++i) {
        // skip non-person objects (should be good enough)
        if ((int)std::round(ids[i]))
            continue;

        if (scores[i] < threshold)
            continue;

        detected++;

        bbox_t bbox = {
            .ymin  = std::max(0, static_cast<int>(bboxes[4 * i]     * image_size + 0.5f)),
            .xmin  = std::max(0, static_cast<int>(bboxes[4 * i + 1] * image_size + 0.5f)),
            .ymax  = std::max(0, static_cast<int>(bboxes[4 * i + 2] * image_size + 0.5f)),
            .xmax  = std::max(0, static_cast<int>(bboxes[4 * i + 3] * image_size + 0.5f)),
            .score = scores[i] * 100 + 0.5f
        };
        tmp_results.push_back(bbox);
    }

    // nothing to sort with single result
    if (tmp_results.size() <= 1)
        return tmp_results;

    // order indices to the tmp_results from highest to lowest scores
    std::vector<int> indices(tmp_results.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return tmp_results[lhs].score > tmp_results[rhs].score;
    });

    while (!indices.empty()) {
        int idx = indices.front();
        indices.erase(indices.begin());
        auto current_box = tmp_results[idx];

        // NMS
        // probably not required as the model has it inside
        // go through remaining box and check if there are any that overlap with the current box and have lower score
        for (auto it = indices.begin(); it != indices.end(); ) {
            //remove index to box that is overlapped or move to next one if not overlapped
            if (IoU(current_box, tmp_results[*it]) > iou_threshold)
                it = indices.erase(it);
            else
                it++;
        }

        // ignore boxes with not enough change
        if (hasChangedAverage(image, last_image, image_size, current_box.xmin, current_box.ymin, current_box.xmax, current_box.ymax, fp_change, fp_count))
            results.push_back(current_box);
        else
            printf("Probably false positive\n");
    }

    //printf("Total results vs final: %d vs %d\n", tmp_results.size(), results.size());
    return results;
}


//takes about 106-108ms
static void detectTask(void *args) {
    status = 0;
    int clean = 0;
    int obj_cnt, ret;
    float det_thresh, iou_thresh, fp_change, fp_count;
    last_frame.resize(image.size(), 0);
    auto detect_image = last_frame;

    //TODO initial "calibration"
    while (true) {
        // get frame
        if (xSemaphoreTake(image_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            ret = CameraTask::GetSingleton()->GetFrame({frame});
            detect_image = image;
            xSemaphoreGive(image_mux);
        }
        else {
            printf("failed to acquire img mutex\n");
            continue;
        }

        if (!ret) {
            printf("Failed to get image from camera.\r\n");
            setStatus(-1);
            continue;
        }

        //copy image data to tensor
        std::memcpy(tflite::GetTensorData<int8_t>(interpreter->input_tensor(0)), detect_image.data(), detect_image.size());

        // run model
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            setStatus(-2);
            continue;
        }

        if (interpreter->outputs().size() != 4) {
            printf("Output size mismatch\\n");
            setStatus(-3);
            continue ;
        }


        // get results and remove anything that is not a person
        if (xSemaphoreTake(config_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            det_thresh = config.det_thresh / 100.0f;
            iou_thresh = config.iou_thresh / 100.0f;
            fp_change = config.fp_change   / 100.0f;
            fp_count = config.fp_count     / 100.0f;
            xSemaphoreGive(config_mux);
        }
        else {
            printf("failed to acquire config mutex\n");
            setStatus(-5);
            continue;
        }

        obj_cnt = 0;
        if (xSemaphoreTake(result_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            results = getResults(interpreter, detect_image, last_frame, image_size, det_thresh, iou_thresh, fp_change, fp_count);
            obj_cnt = results.size();
            xSemaphoreGive(result_mux);
            printf("%d objects\n", obj_cnt);
        }
        else {
            printf("failed to acquire res mutex\n");
            setStatus(-4);
            continue;
        }

        detected ? clean = 0 : clean++;

        if (!detected && clean >= 10) {
            if (xSemaphoreTake(image_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
                last_frame = image;
                xSemaphoreGive(image_mux);
            }
            else {
                printf("failed to acquire img mutex\n");
                setStatus(-5);
                continue;
            }
            printf("Last frame updated\n");
        }

        //printf("%d person objects\n", obj_cnt);
        if (xSemaphoreTake(status_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            status = obj_cnt;
            xSemaphoreGive(status_mux);
        }
        else
            printf("failed to acquire det mutex\n");
    }
}

void setup() {
    status_mux = xSemaphoreCreateMutex();
    image_mux  = xSemaphoreCreateMutex();
    result_mux = xSemaphoreCreateMutex();
    config_mux = xSemaphoreCreateMutex();

    auto input_tensor = interpreter->input_tensor(0);
    if (input_tensor->dims->data[1] != input_tensor->dims->data[2]) {
        printf("Model input is not square. Exiting...\n");
        vTaskDelete(NULL);
    }
    image_size  = input_tensor->dims->data[1];
    printf("Input tensor size: %d %d\n", image_size, image_size);

    // create and save image storages
    image.resize(image_size * image_size * CameraFormatBpp(CameraFormat::kRgb), 0);
    printf("%d\n", image.capacity());

    // create frames for model and preview image
    frame = CameraFrameFormat{
        CameraFormat::kRgb,
        CameraFilterMethod::kBilinear,
        CameraRotation::k270,
        image_size,
        image_size,
        true,
        image.data(),
        true
    };
}

/*----(WEB HADNLING)----*/
HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "index.html"))
        return std::string(MAIN_WEB_PATH);
    
    else if (StrEndsWith(uri, STREAM_WEB_PATH)) {
        imageVector_t img_copy;
        if (xSemaphoreTake(image_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            img_copy = image;
            xSemaphoreGive(image_mux);
        }
        else
            printf("failed to get img mutex\n");

        // get result copy
        //std::vector<tensorflow::Object> res_copy;
        std::vector<bbox_t> res_copy;
        if (xSemaphoreTake(result_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            res_copy = results;
            xSemaphoreGive(result_mux);
        }
        else
            printf("failed to get res mutex\n");

        int quality = DEFAULT_JPEG_QUALITY;
        int mask_size = DEFAULT_MASK_SIZE;
        std::vector<uint8_t> mask;
        if (xSemaphoreTake(config_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            quality   = config.jpeg_quality;
            mask      = config.mask;
            mask_size = config.mask_size;
            xSemaphoreGive(config_mux);
        }
        else
            printf("failed to acquire config mutex\n");

        //TODO move this to detector task
        bool is_unmasked = false;
        std::vector<bbox_t> final_result;
        for (auto result : res_copy) {
            int xmin = result.xmin;
            int xmax = result.xmax;
            int ymin = result.ymin;
            int ymax = result.ymax;

            int ticks = xTaskGetTickCount();
            if (!isMasked(xmin, xmax, ymin, ymax, image_size, mask, mask_size)) {
                is_unmasked = true;
                final_result.push_back(result);
            }
            else
                drawRectangleFast(xmin, ymin, xmax, ymax, &img_copy, image_size, 0, 0, 255);
        }
        detected_buf.pop_front();
        detected_buf.push_back(is_unmasked);

        int detect_cnt = 0;
        for (auto was_detected : detected_buf)
            detect_cnt += was_detected ? 1 : 0;

        printf("%d %d\n", detect_cnt, detected_buf.size());

        for (auto result : final_result) {
            int xmin = result.xmin;
            int xmax = result.xmax;
            int ymin = result.ymin;
            int ymax = result.ymax;

            if (detect_cnt >= 3)
                drawRectangleFast(xmin, ymin, xmax, ymax, &img_copy, image_size, 0, 255, 0);
            else
                drawRectangleFast(xmin, ymin, xmax, ymax, &img_copy, image_size, 255, 255, 0);
        }

        imageVector_t jpeg;
        JpegCompressRgb(img_copy.data(), image_size, image_size, quality, &jpeg);
        return jpeg;
    }

    else if (StrEndsWith(uri, CONFIG_LOAD_WEB_PATH)) {
        std::string cfg_csv;
        int ret = loadConfigFile(&cfg_csv);
        if (ret) {
            printf("Load failed\n");
            return {};
        }
        return std::vector<uint8_t>(cfg_csv.begin(), cfg_csv.end());
    }
    else if (StrEndsWith(uri, CONFIG_RESET_WEB_PATH)) {
        cfg_struct_t default_config;
        std::string cfg_csv = buildConfigCsv(default_config);
        int ret = saveConfigFile(cfg_csv);
        if (ret) {
            printf("Failed to save default config\n");
            return {};
        }
        return std::vector<uint8_t>(cfg_csv.begin(), cfg_csv.end());
    }

    return {};
}

std::string postUriHandler(std::string uri, std::vector<uint8_t> payload) {
    printf("on post finished\r\n");

    if (StrEndsWith(uri, CONFIG_SAVE_WEB_PATH)) {
        int ret = saveConfigFile(std::string(payload.begin(), payload.end()));
        if (ret) {
            printf("Failed to save config: %d\n", ret);
            return {};
        }

        return "/200.html";
    }

    printf("unhandled uri: %s\n", uri.c_str());
    return {};
}



extern "C" void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    config.mask.resize(config.mask_size * config.mask_size, 0);

    // Try and get an IP
    std::optional<std::string> our_ip_addr;
    printf("Attempting to use ethernet...\r\n");
    CHECK(EthernetInit(true));
    our_ip_addr = EthernetGetIp();

    if (our_ip_addr.has_value()) {
        printf("DHCP succeeded, our IP is %s.\r\n", our_ip_addr.value().c_str());
    } else {
        printf("We didn't get an IP via DHCP, not progressing further.\r\n");
        return;
    }


    tflite::MicroErrorReporter error_reporter;
    //Read the model and checks version.
    std::vector<uint8_t> model_raw;  //entire model is stored in this vector
    if (!LfsReadFile(MODEL_PATH, &model_raw)) {
        TF_LITE_REPORT_ERROR(&error_reporter, "Failed to load model!");
        vTaskSuspend(nullptr);
    }


    //Turn on the TPU and get it's context.
    auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
    if (!tpu_context) {
        TF_LITE_REPORT_ERROR(&error_reporter, "ERROR: Failed to get EdgeTpu context!");
        vTaskSuspend(nullptr);
    }

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
        printf("ERROR: Model must have only one input tensor\r\n");
        vTaskSuspend(nullptr);
    }


    CameraTask::GetSingleton()->SetPower(true);
    CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

    setup();

    std::string cfg;
    int ret = loadConfigFile(&cfg);
    if (ret) {
        printf("Failed to load config: %d\n", ret);
        vTaskDelete(NULL);
    }
    printf("Config loaded\n");


    xTaskCreate(detectTask, "detect", 8192, nullptr, 0, &detect_task_handle);
    PostHttpServer http_server;
    http_server.registerPostUriHandler(postUriHandler);
    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);
    
    while(true) {
        printf("ALIVE\n");
        vTaskDelay(2000);
    }

    vTaskSuspend(nullptr);
}