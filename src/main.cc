//TODO separate load and save for config
//TODO configurable grid size
//TODO rotation
//TODO save to file
//TODO load from file
//TODO reset config

//TODO move stuff to m4 core
//TODO move stuff to own classes

//TODO UI???

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

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

#define IDX(x, y, width) (y * width + x) * 3;

#define TENSOR_ARENA_SIZE     8 * 1024 * 1024
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);


int image_size   = 320;
typedef struct {
    int mask_size = 32;
    std::vector<uint8_t> mask;
    int rotation = 270;
    int det_thresh = 50;
    int iou_thresh = 50;
    int fp_change = 40;
    int fp_count = 5;
    int jpeg_quality = 75;
} cfg_t;
cfg_t config;


tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter;
tflite::MicroErrorReporter error_reporter;

std::vector<uint8_t> image;
CameraFrameFormat frame;

SemaphoreHandle_t status_mux;
SemaphoreHandle_t image_mux;
SemaphoreHandle_t result_mux;
SemaphoreHandle_t config_mux;
int status;

TaskHandle_t detect_task_handle; // detector task handle

std::vector<uint8_t> last_frame; // last empty frame before any detection happens
int detected = 0;


typedef struct {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
    float score;
} bbox_t;
std::vector<bbox_t> results;


/** @brief build config csv from current config 
 * 
 * @param config 
 * @return std::string 
 */
std::string buildConfigCsv(cfg_t config) {
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

    printf("Newly built config: %s\n", cfg.c_str());

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
    cfg_t new_config;
    new_config.mask.resize(new_config.mask_size * new_config.mask_size, 0);

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
        return 1;

    //save it to config variable
    return saveConfig(config_csv) ? 2 : 0;
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


void drawHorizontalLine(int y, int x_min, int x_max, std::vector<uint8_t> *image, int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    int start = width * y + x_min;
    for (int i = start; i < start + (x_max - x_min); i++) {
        (*image)[i * 3]     = r;
        (*image)[i * 3 + 1] = g;
        (*image)[i * 3 + 2] = b;
    }
}

void drawVerticalLine(int x, int y_min, int y_max, std::vector<uint8_t> *image, int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    int start_x = y_min * width + x;
    for (int i = start_x; i < start_x + width * (y_max - y_min); i += width) {
        (*image)[i * 3]     = r;
        (*image)[i * 3 + 1] = g;
        (*image)[i * 3 + 2] = b;
    }
}

//TODO make more efficient version
void drawRectangle(int y_min, int y_max, int x_min, int x_max, std::vector<uint8_t> *image, int width, int height, uint8_t r = 0, uint8_t g = 255, uint8_t b = 0) {
    if (y_min > height)
        y_min = height;
    if (y_max < 0)
        y_max = 0;
    if (x_min < 0)
        x_min = 0;
    if (x_max > width)
        x_max = width;
    drawHorizontalLine(y_min, x_min, x_max, image, width, height, r, g, b);
    drawHorizontalLine(y_max, x_min, x_max, image, width, height, r, g, b);
    drawVerticalLine(x_min, y_min, y_max, image, width, height, r, g, b);
    drawVerticalLine(x_max, y_min, y_max, image, width, height, r, g, b);
}


// Helper function to check if the coordinates are inside the box.
inline bool isInsideBox(int x, int y, int x_min, int x_max, int y_min, int y_max) {
    return (x >= x_min && x <= x_max && y >= y_min && y <= y_max);
}

bool isMasked(int x_min, int x_max, int y_min, int y_max, int image_size, const std::vector<uint8_t>& mask, int mask_size) {
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
bool hasChangedAverage(const std::vector<unsigned char>& currentImage, const std::vector<unsigned char>& previousImage, int width, int height, int xmin, int ymin, int xmax, int ymax, float change, float count) {
    int32_t changed = 0;
    int32_t pixel_cnt = (xmax - xmin + 1) * (ymax - ymin + 1);
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            size_t index = IDX(x, y, width);
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


//float IoU(const tensorflow::BBox<float>& a, const tensorflow::BBox<float>& b) {
float IoU(const bbox_t& a, const bbox_t& b) {
    float intersectionArea = std::max(0.0f, std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin)) *
                             std::max(0.0f, std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin));
    float unionArea = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                      (b.xmax - b.xmin) * (b.ymax - b.ymin) - intersectionArea;
    return intersectionArea / unionArea;
}

//std::vector<tensorflow::Object> getResults(tflite::MicroInterpreter *interpreter, float threshold, float iou_threshold) {
std::vector<bbox_t> getResults(tflite::MicroInterpreter *interpreter, float threshold, float iou_threshold, float fp_change, float fp_count) {
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
            .ymin  = std::max(0.0f, bboxes[4 * i]),
            .xmin  = std::max(0.0f, bboxes[4 * i + 1]),
            .ymax  = std::max(0.0f, bboxes[4 * i + 2]),
            .xmax  = std::max(0.0f, bboxes[4 * i + 3]),
            .score = scores[i]
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
            //if (IoU(current_box.bbox, tmp_results[*it].bbox) > iou_threshold)
            if (IoU(current_box, tmp_results[*it]) > iou_threshold)
                it = indices.erase(it);
            else
                it++;
        }

        // ignore boxes with not enough change
        if (xSemaphoreTake(image_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            if (hasChangedAverage(image, last_frame, image_size, image_size, current_box.xmin * image_size, current_box.ymin * image_size, current_box.xmax * image_size, current_box.ymax * image_size, fp_change, fp_count))
                results.push_back(current_box);
            else
                printf("Probably false positive\n");
                xSemaphoreGive(image_mux);
            }
        else {
            printf("failed to acquire img mutex\n");
            setStatus(-80);
            continue;
        }
    }

    //printf("Total results vs final: %d vs %d\n", tmp_results.size(), results.size());
    return results;
}


//takes about 106-108ms
static void detectTask(void *args) {
    status = 0;
    int clean = 0;
    int obj_cnt, det_thresh, iou_thresh, fp_change, fp_count, ret;
    last_frame.resize(image.size(), 0);

    //TODO initial "calibration"
    while (true) {
        // get frame
        if (xSemaphoreTake(image_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            ret = CameraTask::GetSingleton()->GetFrame({frame});
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
        if (xSemaphoreTake(image_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            std::memcpy(tflite::GetTensorData<int8_t>(interpreter->input_tensor(0)), image.data(), image.size());
            xSemaphoreGive(image_mux);
        }
        else {
            printf("failed to acquire img mutex\n");
            continue;
        }

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
            det_thresh = config.det_thresh;
            iou_thresh = config.iou_thresh;
            fp_change = config.fp_change;
            fp_count = config.fp_count;
            xSemaphoreGive(config_mux);
        }
        else {
            printf("failed to acquire config mutex\n");
            setStatus(-5);
            continue;
        }

        obj_cnt = 0;
        if (xSemaphoreTake(result_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            results = getResults(interpreter, det_thresh / 100.0f, iou_thresh / 100.0f, fp_change / 100.0f, fp_count / 100.0f);
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

HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "index.html"))
        return std::string(MAIN_WEB_PATH);
    
    else if (StrEndsWith(uri, STREAM_WEB_PATH)) {
        // get status
        //if (xSemaphoreTake(status_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
        //    printf("Status: %d\n", status);
        //    xSemaphoreGive(status_mux);
        //}
        //else
        //    printf("failed to acquire det mutex\n");

        // get image copy
        std::vector<uint8_t> img_copy;
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

        int quality = 75;
        int mask_size = 32; //TODO macro for contants
        std::vector<uint8_t> mask;
        if (xSemaphoreTake(config_mux, pdMS_TO_TICKS(100)) == pdTRUE) {
            quality   = config.jpeg_quality;
            mask      = config.mask;
            mask_size = config.mask_size;
            xSemaphoreGive(config_mux);
        }
        else
            printf("failed to acquire config mutex\n");

        for (auto result : res_copy) {
            //printf("%f %f %f %f\n", result.bbox.xmin, result.bbox.xmax, result.bbox.ymin, result.bbox.ymax);
            int xmin = result.xmin * image_size;
            int xmax = result.xmax * image_size;
            int ymin = result.ymin * image_size;
            int ymax = result.ymax * image_size;

            


            if (isMasked(xmin, xmax, ymin, ymax, image_size, mask, mask_size))
                drawRectangle(ymin, ymax, xmin, xmax, &img_copy, image_size, image_size, 0, 0, 255);
            else
                drawRectangle(ymin, ymax, xmin, xmax, &img_copy, image_size, image_size, 0, 255, 0);
        }

        std::vector<uint8_t> jpeg;
        JpegCompressRgb(img_copy.data(), image_size, image_size, quality, &jpeg);
        return jpeg;
    }

    else if (StrEndsWith(uri, CONFIG_LOAD_WEB_PATH)) {
        std::string str;
        int ret = loadConfigFile(&str);
        if (ret) {
            printf("Load failed\n");
            return {};
        }
        return std::vector<uint8_t>(str.begin(), str.end());
    }
    else if (StrEndsWith(uri, CONFIG_RESET_WEB_PATH)) {
        printf("Load not implemented yet");
        return {};
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

        return "/index.html";
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