/**
 * @file main.cpp
 * @author Lukáš Baštýř (l.bastyr@seznam.cz, 492875@muni.cz)
 * @brief A program for person presence detection in configurable area
 * @date 10-11-2023
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <deque>
#include <cstdarg>
#include <cstdlib>

#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/gpio.h"
#include "libs/base/ethernet.h"
#include "libs/base/watchdog.h"
#include "libs/base/reset.h"
#include "libs/base/mutex.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_op.h"
#include "libs/tpu/edgetpu_manager.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "post_http_server.hpp"
#include "ConfigManager.hpp"


using namespace coralmicro;
using namespace std;


/*----(FUNCTION MACROS)----*/
#define IMAGE_SIZE gl_image_size * gl_image_size * 3

/*----(WEB RESPONSES)----*/
#define MAIN_PAGE_RESPONSE "/page.html"
#define OK_RESPONSE        "/ok.txt"

/*----(WEB PATHS)----*/
#define STREAM_WEB_PATH           "/camera_stream"
#define DETECTION_STATUS_WEB_PATH "/detection_status"
#define CONFIG_SAVE_WEB_PATH      "/config_save"
#define CONFIG_LOAD_WEB_PATH      "/config_load"
#define CONFIG_RESET_WEB_PATH     "/config_reset"
#define OK_WEB_PATH               "/ok"

/*----(LOCAL FILES)----*/
#define MODEL_PATH  "/model.tflite"
#define CONFIG_PATH "/cfg.bin"

#define PIN0 kSpiCs
#define PIN1 kSpiSck

#define BBOX_SMALL    1
#define BBOX_MASKED   2
#define BBOX_UNSURE   3
#define BBOX_UNMASKED 4

/*----(DEFAULT VALUES)----*/
#define DEFAULT_IMAGE_SIZE   320
#define DEFAULT_MASK_SIZE    32
#define DEFAULT_ROTATION     0
#define DEFAULT_DET_THRESH   25
#define DEFAULT_IOU_THRESH   25
#define DEFAULT_FP_CHANGE    0
#define DEFAULT_FP_COUNT     0
#define DEFAULT_MASK_THRESH  70
#define DEFAULT_MIN_WIDTH    0
#define DEFAULT_MIN_HEIGHT   0
#define DEFAULT_MIN_AS_AREA  false

/*----(TENSOR ARENA)----*/
#define TENSOR_ARENA_SIZE 5700000 //enough from testing
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE); //OCRAM is too small for this


/*----(STRUCTS)----*/
typedef std::vector<uint8_t> image_vector_t;

typedef struct {
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int score;
    int type;
} bbox_t;
typedef std::vector<bbox_t> bbox_vector_t;


/*----(TFLITE MICRO)----*/
tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter;
tflite::MicroErrorReporter error_reporter;

/*----(FREERTOS)----*/
SemaphoreHandle_t data_mux;
SemaphoreHandle_t sync_sem;
TaskHandle_t detect_task_handle;

/*----(SHARED GLOBAL VARIABLES)----*/
int gl_status          = 0;
int gl_image_size      = DEFAULT_IMAGE_SIZE;
int gl_inference_time  = 0;
int gl_camera_time     = 0;
int gl_postprocess_time = 0;
bool gl_update_config  = true;
//cfg_struct_t gl_config;
image_vector_t gl_image;
bbox_vector_t gl_bboxes;

ConfigManager cfg_manager;



/*----(HELPERS)----*/
void exit();
void reset();


/*----(GET & POST HANDLERS)----*/
HttpServer::Content getUriHandler(const char* uri);

string postUriHandler(string uri, vector<uint8_t> payload);


/*----(CONFIG HANDLING FUNCTIONS)----*/
cfg_struct_t buildDefaultConfig();

bool configFromRaw(vector<uint8_t> raw_data, cfg_struct_t *config);

vector<uint8_t> configToRaw(const cfg_struct_t &config);

bool writeConfigToFile(const char *path, const vector<uint8_t> &raw_cfg);

vector<uint8_t> readConfigFromFile(const char *path);


/*----(DETECTION TASK FUNCTIONS)----*/
static void detectTask(void *args);

int postprocessResults(vector<bbox_t> *results, tflite::MicroInterpreter *interpreter,
                       const image_vector_t &image, const image_vector_t &background, const cfg_struct_t &config);

float IoU(const bbox_t& a, const bbox_t& b);

bool hasBBoxChanged(const image_vector_t &current_image, const image_vector_t &previous_image,
                    int image_size, bbox_t bbox, float change, float count);

bool isBBoxMasked(bbox_t bbox, const image_vector_t& mask, int mask_size, int image_size, float threshold);

bool isInsideBox(int x, int y, int xmin, int xmax, int ymin, int ymax);




/*----(MAIN ENTRYPOINT)----*/
extern "C" [[noreturn]] void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    GpioSetMode(PIN0, GpioMode::kOutput);
    GpioSetMode(PIN1, GpioMode::kOutput);


    data_mux   = xSemaphoreCreateMutex();
    sync_sem   = xSemaphoreCreateBinary();


    auto reset_stats = ResetGetStats();
    printf("Reset stats:\n");
    printf("  Reason:     %ld\n", reset_stats.reset_reason);
    printf("  wdog cnt:   %ld\n", reset_stats.watchdog_resets);
    printf("  lockup cnt: %ld\n", reset_stats.lockup_resets);


    WatchdogStart({
        .timeout_s = 2,
        .pet_rate_s = 1,
        .enable_irq = false
    });


    //load config
    int ret = cfg_manager.load(CONFIG_PATH);
    switch (ret) {
    case 2:
        printf("Stored config is invalid\n");
    case 1:
        cfg_manager.init();
        if (cfg_manager.save(CONFIG_PATH))
            printf("Default config saved\n");
        else {
            printf("Failed to save default config\n");
            reset();
        }
        break;
    default:
        printf("Config loaded from file\n");
        break;
    }

    // Try and get an IP
    printf("Attempting to use ethernet...\n");
    bool eth_succ = EthernetInit(true);
    if (!eth_succ) {
        printf("Failed to init ethernet\n");
        reset();
    }
    else {
        auto ip_addr = EthernetGetIp();
        if (ip_addr.has_value())
            printf("DHCP succeeded, our IP is %s\n", ip_addr.value().c_str());
        else {
            printf("We didn't get an IP via DHCP. Continuing wihtout internet.\n");
            eth_succ = false;
        }
    }


    //Read the model and checks version.
    vector<uint8_t> model_raw;  //entire model is stored in this vector
    if (!LfsReadFile(MODEL_PATH, &model_raw)) {
        printf("Failed to load model\n");
        exit();
    }

    //confirm model version
    auto* model = tflite::GetModel(model_raw.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model version: %ld\nSupported version: %d\n", model->version(), TFLITE_SCHEMA_VERSION);
        exit();
    }

    //create micro interpreter
    resolver.AddDequantize();
    resolver.AddDetectionPostprocess();
    resolver.AddCustom(kCustomOp, RegisterCustomOp());
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &error_reporter);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Failed to allocate tensors\n");
        exit();
    }

    if (interpreter->inputs().size() != 1) {
        printf("Model must have only one input tensor\n");
        exit();
    }

    auto input_tensor = interpreter->input_tensor(0);
    if (input_tensor->dims->data[1] != input_tensor->dims->data[2]) {
        printf("Model input is not square\n");
        exit();
    }

    if (interpreter->outputs().size() != 4) {
        printf("Invalid output tensor count\n");
        exit();
    }

    //Turn on the TPU and get it's context.
    auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
    if (!tpu_context) {
        printf("Failed to get TPU context\n");
        exit();
    }

    //start camera
    CameraTask::GetSingleton()->SetPower(false);
    vTaskDelay(100);
    CameraTask::GetSingleton()->SetPower(true);
    CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

    //start detection task
    xTaskCreate(detectTask, "detect", 16000, nullptr, 4, &detect_task_handle);
    xSemaphoreTake(sync_sem, portMAX_DELAY);

    //skip http server if no internet
    if (!eth_succ)
        vTaskSuspend(nullptr);

    //start http server
    PostHttpServer http_server;
    http_server.AddUriHandler(getUriHandler);
    http_server.AddPostUriHandler(postUriHandler);
    UseHttpServer(&http_server);

    vTaskSuspend(nullptr);
}


/*----(HELPERS)----*/
void exit() {
    printf("Exiting...\n");
    LedSet(Led::kStatus, false);
    vTaskSuspend(NULL);
}

void reset() {
    printf("Restarting...\n");
    LedSet(Led::kStatus, false);
    vTaskDelay(500);
    ResetToFlash();
}


/*----(GET & POST HANDLERS)----*/
/** @brief GET request handler
 * 
 * @param uri URI of the GET request
 * @return HttpServer::Content
 */
HttpServer::Content getUriHandler(const char* uri) {
    //root
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "index.html"))
        return string(MAIN_PAGE_RESPONSE);

    //image and results
    else if (StrEndsWith(uri, STREAM_WEB_PATH)) {
        xSemaphoreTake(sync_sem, portMAX_DELAY);
        MutexLock lock(data_mux);

        //create packet with status, results and image and send it
        vector<uint8_t> packet;
        packet.push_back(gl_status + 128);
        packet.push_back(gl_camera_time);
        packet.push_back(gl_inference_time);
        packet.push_back(gl_postprocess_time);
        packet.push_back(gl_bboxes.size());
        for (auto bbox : gl_bboxes) {
            packet.push_back(bbox.score);
            packet.push_back(bbox.type);
            packet.push_back(bbox.xmax >> 8);
            packet.push_back(bbox.xmax);
            packet.push_back(bbox.ymax >> 8);
            packet.push_back(bbox.ymax);
            packet.push_back(bbox.xmin >> 8);
            packet.push_back(bbox.xmin);
            packet.push_back(bbox.ymin >> 8);
            packet.push_back(bbox.ymin);
        }
        packet.push_back(gl_image_size >> 8);
        packet.push_back(gl_image_size);
        packet.insert(packet.end(), gl_image.begin(), gl_image.end());

        return packet;
    }

    //detection only
    else if (StrEndsWith(uri, DETECTION_STATUS_WEB_PATH)) {
        vector<uint8_t> status;
        status.push_back(gl_status);
        return status;
    }

    //load config from device
    else if (StrEndsWith(uri, CONFIG_LOAD_WEB_PATH)) {
        return cfg_manager.toBinary();
    }

    //reset config
    else if (StrEndsWith(uri, CONFIG_RESET_WEB_PATH)) {
        cfg_manager.init();
        gl_update_config = true;
        if (!cfg_manager.save(CONFIG_PATH))
            printf("Failed to save config to file\n");
        
        return cfg_manager.toBinary();
    }

    //POST OK
    else if (StrEndsWith(uri, OK_WEB_PATH)) {
        return string(OK_RESPONSE);
    }

    return {};
}

/** @brief POST request handler
 * 
 * @param uri URI of the POST request
 * @param payload payload of the POST request
 * @return string - URI path of OK file on success. Empty otherwise.
 */
string postUriHandler(string uri, vector<uint8_t> payload) {
    //only config save is handled
    if (!StrEndsWith(uri, CONFIG_SAVE_WEB_PATH))
        return {};

    if (!cfg_manager.fromBinary(payload))
        return {};

    gl_update_config = true;

    if (cfg_manager.save(CONFIG_PATH))
        return OK_WEB_PATH;

    return {};
}


/*----(DETECTION TASK FUNCTIONS)----*/
/** @brief Main detection task*/
static void detectTask(void *args) {
    gl_status = 0;                          //set status
    cfg_struct_t config;
    cfg_manager.getConfigCopy(&config);        //copy current config
    gl_image.resize(IMAGE_SIZE);            //resize image
    image_vector_t background(IMAGE_SIZE);  //create background

    int bckg_upd_cnt = 0;

    //use input tensor as direct image storage
    uint8_t *tensor_input = tflite::GetTensorData<uint8_t>(interpreter->input_tensor(0));

    //create default frame
    CameraFrameFormat frame = {
        CameraFormat::kRgb,
        CameraFilterMethod::kNearestNeighbor,
        CameraRotation::k0,
        gl_image_size,
        gl_image_size,
        true,
        tensor_input,
        true
    };

    xSemaphoreGive(sync_sem);   //sync with main task

    TickType_t last_wake_time;
    const TickType_t frequency = 50;

    int inference_time = 0;

    while (true) {
        vTaskDelayUntil(&last_wake_time, frequency);

        //update config
        if (gl_update_config) {
            cfg_manager.getConfigCopy(&config);
            gl_update_config = false;
            frame.rotation = static_cast<coralmicro::CameraRotation>(config.rotation);
        }

        //get last image from camera
        int ticks = xTaskGetTickCount();
        int ret = CameraTask::GetSingleton()->GetFrame({frame});
        if (!ret) {
            printf("Failed to get image from camera.\n");
            gl_status = -1;
            xSemaphoreGive(sync_sem);
            reset();
        }
        gl_camera_time = xTaskGetTickCount() - ticks;

        //run model
        int inference_ticks = xTaskGetTickCount();
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            gl_status = -2;
            xSemaphoreGive(sync_sem);
            reset();
        }
        gl_inference_time = xTaskGetTickCount() - inference_ticks;

        //get results from inference and update global variables
        xSemaphoreTake(data_mux, portMAX_DELAY);
        int extract_time = xTaskGetTickCount();
        gl_image = image_vector_t(tensor_input, tensor_input + IMAGE_SIZE);
        gl_status = postprocessResults(&gl_bboxes, interpreter, gl_image, background, config);
        gl_postprocess_time = xTaskGetTickCount() - extract_time;
        xSemaphoreGive(data_mux);
        xSemaphoreGive(sync_sem);   //sync with web task (if it is waiting for us)

        //toggle led and pins
        LedSet(Led::kUser, gl_status > 2);
        GpioSet(PIN0, gl_status < 3 && gl_status > 0);
        GpioSet(PIN1, gl_status > 2);

        //update background
        bckg_upd_cnt = gl_status ? 0 : bckg_upd_cnt + 1;
        if (bckg_upd_cnt >= 10)
            background = gl_image;

        inference_time = (inference_time + xTaskGetTickCount() - ticks) / 2;
        printf("Average detection time: %03d ms | Status: %d\n", inference_time, gl_status);
    }
}

/** @brief Get results from inference.
 * 1. get all bboxes with person id
 * 2. NMS
 * 3. Check which boxes are masked
 * 3. Calculate change against background
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
int postprocessResults(vector<bbox_t> *results, tflite::MicroInterpreter *interpreter,
                       const image_vector_t &image, const image_vector_t &background, const cfg_struct_t &config) {

    static int consec_unmasked = 0; //keeps count of consecutive unmasked detections

    float det_thresh  = config.det_thresh  / 100.0f;
    float iou_thresh  = config.iou_thresh  / 100.0f;
    float fp_change   = config.fp_change   / 100.0f;
    float fp_count    = config.fp_count    / 100.0f;
    float mask_thresh = config.mask_thresh / 100.0f;

    //get data from output tensor
    float *bboxes = tflite::GetTensorData<float>(interpreter->output_tensor(0));
    float *ids    = tflite::GetTensorData<float>(interpreter->output_tensor(1));
    float *scores = tflite::GetTensorData<float>(interpreter->output_tensor(2));
    float *count  = tflite::GetTensorData<float>(interpreter->output_tensor(3));

    int ret = 0;
    bbox_vector_t tmp_results;
    results->clear();
    int cnt = static_cast<int>(count[0]);
    for (int i = 0; i < cnt; ++i) {
        //skip object with id != 0 (not person) and below threshold
        if (ids[i] >= 1 || scores[i] < det_thresh)
            continue;

        //create bbox
        bbox_t bbox = {
            .xmin  = static_cast<int>(max(0.0f, round(bboxes[4 * i + 1] * gl_image_size))),
            .ymin  = static_cast<int>(max(0.0f, round(bboxes[4 * i]     * gl_image_size))),
            .xmax  = static_cast<int>(max(0.0f, round(bboxes[4 * i + 3] * gl_image_size))),
            .ymax  = static_cast<int>(max(0.0f, round(bboxes[4 * i + 2] * gl_image_size))),
            .score = static_cast<int>(round(scores[i] * 100)),
            .type  = BBOX_MASKED
        };
        if (bbox.xmax >= gl_image_size)
            bbox.xmax = gl_image_size - 1;
        if (bbox.ymax >= gl_image_size)
            bbox.ymax = gl_image_size - 1;

        //check dimensions
        if (config.min_as_area) {
            if (config.min_width * config.min_height > (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)) {
                bbox.type = BBOX_SMALL;
                ret       = BBOX_SMALL;
            }
        }
        else {
            if (config.min_width > (bbox.xmax - bbox.xmin) || config.min_height > (bbox.ymax - bbox.ymin)) {
                bbox.type = BBOX_SMALL;
                ret       = BBOX_SMALL;
            }
        }

        tmp_results.push_back(bbox);
    }

    //nothing valid detected
    if (!tmp_results.size()) {
        results->clear();
        if (consec_unmasked > 0)
            consec_unmasked--;
        ret = consec_unmasked > 0 ? BBOX_UNSURE : 0;
        return ret; //>0 || nothing
    }

    //single object detected
    if (tmp_results.size() == 1) {
        //change type if unmasked
        if (tmp_results[0].type != BBOX_SMALL && !isBBoxMasked(tmp_results[0], config.mask, config.mask_size, gl_image_size, mask_thresh)) {
            if (consec_unmasked < 5)
                consec_unmasked++;
            tmp_results[0].type = consec_unmasked >= 3 ? BBOX_UNMASKED : BBOX_UNSURE; //>=3/5 || > 0
            *results = tmp_results;
            return tmp_results[0].type;
        }

        //decrease if masked
        if (consec_unmasked > 0)
            consec_unmasked--;
        *results = tmp_results;
        //if big enough
        if (!ret)
            ret = consec_unmasked > 0 ? BBOX_UNSURE : BBOX_MASKED; //>0 || masked
        return ret;
    }

    //order indices to the tmp_results from highest to lowest scores
    vector<int> indices(tmp_results.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return tmp_results[lhs].score > tmp_results[rhs].score;
    });

    bool once = false;
    while (!indices.empty()) {
        int idx = indices.front();
        indices.erase(indices.begin());
        auto current_box = tmp_results[idx];

        //NMS
        //go through remaining box and check if there are any that overlap with the current box and have lower score
        for (auto it = indices.begin(); it != indices.end();) {
            //remove index to box that is overlapped or move to next one if not overlapped
            if (IoU(current_box, tmp_results[*it]) > iou_thresh)
                it = indices.erase(it);
            else
                it++;
        }

        //ignore boxes with not enough change
        if (!hasBBoxChanged(image, background, gl_image_size, current_box, fp_change, fp_count))
            continue;

        //check if box is not masked and update consecutive detection counter and pick right type
        if (current_box.type != BBOX_SMALL && !isBBoxMasked(current_box, config.mask, config.mask_size, gl_image_size, mask_thresh)) {
            //unmasked box -> increase conscutive detection (but only once, since it is per frame)
            if (!once) {
                once = true;
                if (consec_unmasked < 5)
                    consec_unmasked++;
                ret = consec_unmasked >= 3 ? BBOX_UNMASKED : BBOX_UNSURE;
            }
            current_box.type = consec_unmasked >= 3 ? BBOX_UNMASKED : BBOX_UNSURE; //>= 3/5 || > 0
        }
        results->push_back(current_box);
    }

    //all boxes were masked -> decrease
    if (!once && consec_unmasked > 0)
        consec_unmasked--;

    //either nothing, too small or masked
    if (ret < BBOX_UNSURE)
        ret = consec_unmasked ? BBOX_MASKED : ret;
    return ret;
}

/** @brief Intersection over union calculations for overlaping bounding boxes
 * 
 * @param a first bounding box
 * @param b second boundig box
 * @return overlap ratio
 */
float IoU(const bbox_t& a, const bbox_t& b) {
    float int_area = max(0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin)) *
                     max(0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin));
    float un_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                    (b.xmax - b.xmin) * (b.ymax - b.ymin) - int_area;
    return int_area / un_area;
}

/** @brief Function to check if the specified region has changed significantly - average color approach
 * 
 * @param current_image 
 * @param previous_image 
 * @param image_size image dimensions
 * @param bbox bounding box
 * @param change how much individual pixels need to change to be counted
 * @param count how many pixels have to change
 * @return true if changed enough, false otherwise
 */
bool hasBBoxChanged(const image_vector_t &current_image, const image_vector_t &previous_image,
                    int image_size,bbox_t bbox, float change, float count) {

    if (!static_cast<int>(count))
        return true;

    int32_t changed = 0;
    int32_t pixel_cnt = (bbox.xmax - bbox.xmin + 1) * (bbox.ymax - bbox.ymin + 1);
    for (int y = bbox.ymin; y <= bbox.ymax; ++y) {
        for (int x = bbox.xmin; x <= bbox.xmax; ++x) {
            size_t index = (y * image_size + x) * 3;
            int pixel_change = abs(current_image[index] - previous_image[index]) +
                               abs(current_image[index + 1] - previous_image[index + 1]) +
                               abs(current_image[index + 2] - previous_image[index + 2]);
            if (pixel_change > 0xFF * 3 * change)
                changed++;
        }
    }

    return changed >= round(pixel_cnt * count);
}

/** @brief Check if enough bbox is masked
 * 
 * @param bbox bounding box
 * @param mask mask
 * @param mask_size mask dimensions
 * @param image_size image dimensions
 * @param threshold mask threshold
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
            int grid_x = x / scale;
            int grid_y = y / scale;

            // Check if the current point is inside the box.
            if (isInsideBox(grid_x, grid_y, bbox.xmin/scale, bbox.xmax/scale, bbox.ymin/scale, bbox.ymax/scale)) {
                // Index in the grid
                int index = grid_y * mask_size + grid_x;

                // Check if the cell is non-toggled
                covered_cells += mask[index];
                total_cells++;
            }
        }
    }

    return covered_cells > total_cells * (1.0f - threshold);
}

/** @brief Helper function to check if the coordinates are inside the box.*/
bool isInsideBox(int x, int y, int xmin, int xmax, int ymin, int ymax) {
    return (x >= xmin && x <= xmax && y >= ymin && y <= ymax);
}
