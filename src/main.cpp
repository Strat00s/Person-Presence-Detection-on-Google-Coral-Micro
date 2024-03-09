/**
 * @file main.cpp
 * @author Lukáš Baštýř (l.bastyr@seznam.cz)
 * @brief 
 * @version 1.0
 * @date 10-11-2023
 * 
 * @copyright Copyright (c) 2024
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

#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/base/gpio.h"
#include "libs/base/filesystem.h"
#include "libs/base/ethernet.h"
#include "libs/base/check.h"
#include "libs/base/watchdog.h"
#include "libs/base/reset.h"
#include "libs/base/mutex.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_op.h"
#include "libs/tpu/edgetpu_manager.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/dns.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/tcpip.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "local_libs/http_server.h"
#include "structs.hpp"


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
#define NEW_CONFIG_PATH "/cfg.bin"

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
#define DEFAULT_JPEG_QUALITY 70
#define DEFAULT_MIN_WIDTH    0
#define DEFAULT_MIN_HEIGHT   0
#define DEFAULT_MIN_AS_AREA  false

/*----(TENSOR ARENA)----*/
#define TENSOR_ARENA_SIZE 5700000 //enough from testing
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE); //OCRAM is too small for this


/*----(TFLITE MICRO)----*/
tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter;
tflite::MicroErrorReporter error_reporter;

/*----(SHARED GLOBAL VARIABLES)----*/
int gl_status          = 0;
int gl_image_size      = DEFAULT_IMAGE_SIZE;
int gl_inference_time  = 0;
int gl_camera_time     = 0;
int gl_postprocess_time = 0;
bool gl_update_config  = true;
cfg_struct_t gl_config;
image_vector_t gl_image;
bbox_vector_t gl_bboxes;

/*----(FREERTOS)----*/
SemaphoreHandle_t config_mux;
SemaphoreHandle_t data_mux;
SemaphoreHandle_t sync_sem;
TaskHandle_t detect_task_handle;


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
    cfg.min_width    = DEFAULT_MIN_WIDTH;
    cfg.min_height   = DEFAULT_MIN_HEIGHT;
    cfg.min_as_area  = DEFAULT_MIN_AS_AREA;
    cfg.mask.resize(cfg.mask_size * cfg.mask_size, 0);
    return cfg;
}


/** @brief Generate config from csv
 * 
 * @param csv 
 * @param config 
 * @return true on success, false on invalid csv (csv has less tha 8 fields)
 */
bool configFromCsv(const string &csv, cfg_struct_t *config) {
    if (count(csv.begin(), csv.end(), ',') < 11) {
        printf("invalid csv received\n");
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
        if (end == string::npos) {
            finished = true;
            end = csv.size();
        }
        string field_val = csv.substr(start, end - start);

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
        val = strtol(field_val.c_str(), &endptr, 10);
        if (*endptr) {
            printf("Failed to convert %s to int\n", field_val.c_str());
            malformed = true;
            continue;
        }

        //everything but mask
        switch (field_num) {
            case 0:  new_config.mask_size    = val; break;
            case 1:  new_config.mask_thresh  = val; break;
            case 3:  new_config.rotation     = val; break;
            case 4:  new_config.det_thresh   = val; break;
            case 5:  new_config.iou_thresh   = val; break;
            case 6:  new_config.fp_change    = val; break;
            case 7:  new_config.fp_count     = val; break;
            case 8:  new_config.jpeg_quality = val; break;
            case 9:  new_config.min_width    = val; break;
            case 10: new_config.min_height   = val; break;
            case 11: new_config.min_as_area  = val; break;
            default: printf("Unhandled field %d: %s\n", field_num, field_val.c_str()); break;
        }
    }

    if (malformed)
        new_config.mask.resize(new_config.mask_size * new_config.mask_size, 0);

    *config = new_config;
    return true;
}

bool configFromRaw(std::vector<uint8_t> raw_data, cfg_struct_t *config) {
    printf("raw_data len: %d\n", raw_data.size());
    //raw config is smaller than minimun size
    if (raw_data.size() < 10 + 4)
        return false;
    
    int mask_size     = raw_data[0] * raw_data[0];
    //reported mask size does not match the raw config size
    if (raw_data.size() != 10 + mask_size)
        return false;

    config->mask_size   = raw_data[0];
    config->mask_thresh = raw_data[1];
    vector<uint8_t>::const_iterator first = raw_data.begin() + 2;
    vector<uint8_t>::const_iterator last  = raw_data.begin() + 2 + mask_size;
    config->mask        = vector<uint8_t>(first, last);
    config->rotation    = raw_data[mask_size + 3];
    config->det_thresh  = raw_data[mask_size + 4];
    config->iou_thresh  = raw_data[mask_size + 5];
    config->fp_change   = raw_data[mask_size + 6];
    config->fp_count    = raw_data[mask_size + 7];
    config->min_width   = raw_data[mask_size + 8];
    config->min_height  = raw_data[mask_size + 9];
    config->min_as_area = raw_data[mask_size + 10];

    return true;
}

/** @brief 
 * 
 * @param config 
 * @return string 
 */
string csvFromConfig(const cfg_struct_t &config) {
    string csv = to_string(config.mask_size) + ',';
    csv += to_string(config.mask_thresh) + ',';
    for (auto val: config.mask)
        csv += to_string(val);
    csv += ',';
    csv += to_string(config.rotation)     + ',';
    csv += to_string(config.det_thresh)   + ',';
    csv += to_string(config.iou_thresh)   + ',';
    csv += to_string(config.fp_change)    + ',';
    csv += to_string(config.fp_count)     + ',';
    csv += to_string(config.jpeg_quality) + ',';
    csv += to_string(config.min_width)    + ',';
    csv += to_string(config.min_height)   + ',';
    csv += config.min_as_area ? '1' : '0';
    return csv;
}

std::vector<uint8_t> configToRaw(const cfg_struct_t &config) {
    std::vector<uint8_t> raw_data;
    raw_data.push_back(config.mask_size);
    raw_data.push_back(config.mask_thresh);
    raw_data.insert(raw_data.end(), config.mask.begin(), config.mask.end());
    raw_data.push_back(config.rotation);
    raw_data.push_back(config.det_thresh);
    raw_data.push_back(config.iou_thresh);
    raw_data.push_back(config.fp_change);
    raw_data.push_back(config.fp_count);
    raw_data.push_back(config.min_width);
    raw_data.push_back(config.min_height);
    raw_data.push_back(config.min_as_area);

    return raw_data;
}

/** @brief Write csv config to file
 * 
 * @param csv 
 * @return true on success, false otherwise
 */
bool writeConfigFile(const string &csv) {
    return LfsWriteFile(CONFIG_PATH, csv);
}

bool writeConfigToFile(const char *path, const std::vector<uint8_t> &raw_cfg) {
    return LfsWriteFile(path, raw_cfg.data(), raw_cfg.size());
}

/** @brief Read csv config from file
 * 
 * @return empty string on failure
 */
string readConfigFile() {
    string config_csv;
    if (!LfsReadFile(CONFIG_PATH, &config_csv))
        return {};
    return config_csv;
}

vector<uint8_t> readConfigFromFile(const char *path) {
    vector<uint8_t> raw_data;
    if (!LfsReadFile(path, &raw_data))
        return {};
    return raw_data;
}


/*----(MASK HANDLING)----*/
// Helper function to check if the coordinates are inside the box.
inline bool isInsideBox(int x, int y, int xmin, int xmax, int ymin, int ymax) {
    return (x >= xmin && x <= xmax && y >= ymin && y <= ymax);
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
            int pixel_change = abs(current_image[index] - previous_image[index]) +
                               abs(current_image[index + 1] - previous_image[index + 1]) +
                               abs(current_image[index + 2] - previous_image[index + 2]);
            if (pixel_change > 0xFF * 3 * change)
                changed++;
        }
    }

    return changed > round(pixel_cnt * count);
}

/** @brief Intersection over union calculations for overlaping bounding boxes
 * 
 * @param a First bounding box
 * @param b Second boundig box
 * @return overlap ratio
 */
float IoU(const bbox_t& a, const bbox_t& b) {
    float int_area = max(0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin)) *
                     max(0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin));
    float un_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                    (b.xmax - b.xmin) * (b.ymax - b.ymin) - int_area;
    return int_area / un_area;
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
    size_t cnt = static_cast<int>(count[0]);
    for (int i = 0; i < cnt; ++i) {
        //skip object with id != 0 (not person) and below threshold
        if (ids[i] >= 1 || scores[i] < det_thresh)
            continue;

        bbox_t bbox = {
            .xmin  = max(0.0f, round(bboxes[4 * i + 1] * gl_image_size)),
            .ymin  = max(0.0f, round(bboxes[4 * i]     * gl_image_size)),
            .xmax  = max(0.0f, round(bboxes[4 * i + 3] * gl_image_size)),
            .ymax  = max(0.0f, round(bboxes[4 * i + 2] * gl_image_size)),
            .score = round(scores[i] * 100),
            .type  = BBOX_MASKED
        };
        if (bbox.xmax >= gl_image_size)
            bbox.xmax = gl_image_size - 1;
        if (bbox.ymax >= gl_image_size)
            bbox.ymax = gl_image_size - 1;

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

    printf("Prelimenary results: %d\n", tmp_results.size());

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

    // order indices to the tmp_results from highest to lowest scores
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



//Main detect task
static void detectTask(void *args) {
    gl_status = 0;
    gl_image.resize(IMAGE_SIZE);

    cfg_struct_t config = gl_config;
    image_vector_t background(IMAGE_SIZE);
    uint8_t *tensor_input = tflite::GetTensorData<uint8_t>(interpreter->input_tensor(0));

    CameraFrameFormat frame = {
        CameraFormat::kRgb,
        CameraFilterMethod::kNearestNeighbor,
        CameraRotation::k270,
        gl_image_size,
        gl_image_size,
        true,
        tensor_input,
        true
    };

    int bckg_upd_cnt = 0;
    int inference_time = 0;

    TickType_t last_wake_time;
    const TickType_t frequency = 50;

    while (true) {
        vTaskDelayUntil(&last_wake_time, frequency);

        int ticks = xTaskGetTickCount();
        int ret = CameraTask::GetSingleton()->GetFrame({frame});
        if (!ret) {
            printf("Failed to get image from camera.\r\n");
            gl_status = -1;
            xSemaphoreGive(sync_sem);
            reset();
        }
        gl_camera_time = xTaskGetTickCount() - ticks;

        // run model
        int inference_ticks = xTaskGetTickCount();
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            gl_status = -2;
            xSemaphoreGive(sync_sem);
            reset();
        }
        gl_inference_time = xTaskGetTickCount() - inference_ticks;

        if (gl_update_config) {
            xSemaphoreTake(config_mux, portMAX_DELAY);
            config = gl_config;
            gl_update_config = false;
            frame.rotation = static_cast<coralmicro::CameraRotation>(gl_config.rotation);
            xSemaphoreGive(config_mux);
        }

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
        if (bckg_upd_cnt >= 10) {
            background = gl_image;
            printf("Background updated\n");
        }

        inference_time = (inference_time + xTaskGetTickCount() - ticks) / 2;
        printf("Average detect time: %d\n", inference_time);
    }
}


/*----(REQUEST HANDLERS)----*/
HttpServer::Content getUriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "index.html"))
        return string(MAIN_PAGE_RESPONSE);

    else if (StrEndsWith(uri, STREAM_WEB_PATH)) {
        xSemaphoreTake(sync_sem, portMAX_DELAY);
        MutexLock lock(data_mux);

        //create packet with status, results and image and send it
        std::vector<uint8_t> packet;
        packet.push_back(gl_status + 128);
        packet.push_back(gl_camera_time);
        packet.push_back(gl_inference_time);
        packet.push_back(gl_postprocess_time);
        packet.push_back(gl_bboxes.size());
        for (auto bbox : gl_bboxes) {
            packet.push_back(bbox.score);
            packet.push_back(bbox.type);
            int i = packet.size();
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

    else if (StrEndsWith(uri, DETECTION_STATUS_WEB_PATH)) {
        vector<uint8_t> status;
        status.push_back(gl_status);
        return status;
    }

    else if (StrEndsWith(uri, CONFIG_LOAD_WEB_PATH)) {
        MutexLock lock(config_mux);
        return configToRaw(gl_config);
    }

    else if (StrEndsWith(uri, CONFIG_RESET_WEB_PATH)) {
        xSemaphoreTake(config_mux, portMAX_DELAY);
        gl_config = buildDefaultConfig();
        gl_update_config = true;
        auto raw_cfg = configToRaw(gl_config);
        xSemaphoreGive(config_mux);

        if (!writeConfigToFile(NEW_CONFIG_PATH, raw_cfg))
            printf("Failed to write reset config to file\n");

        return raw_cfg;
    }

    else if (StrEndsWith(uri, OK_WEB_PATH)) {
        return string(OK_RESPONSE);
    }

    return {};
}

string postUriHandler(string uri, vector<uint8_t> payload) {
    if (StrEndsWith(uri, CONFIG_SAVE_WEB_PATH)) {
        
        xSemaphoreTake(config_mux, portMAX_DELAY);
        if (!configFromRaw(payload, &gl_config)) {
            printf("Invalid config data provided\n");
            xSemaphoreGive(config_mux);
            return {};
        }
        gl_update_config = true;
        xSemaphoreGive(config_mux);

        if (!writeConfigToFile(NEW_CONFIG_PATH, payload)) {
            printf("Failed to write config to file\n");
            return {};
        }

        return string(OK_WEB_PATH);
    }

    printf("unhandled uri: %s\n", uri.c_str());
    return {};
}



extern "C" [[noreturn]] void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    GpioSetMode(PIN0, GpioMode::kOutput);
    GpioSetMode(PIN1, GpioMode::kOutput);


    config_mux = xSemaphoreCreateMutex();
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
    auto raw_cfg = readConfigFromFile(NEW_CONFIG_PATH);
    //empty/non-existant file or malformed csv
    if (!raw_cfg.size() || !configFromRaw(raw_cfg, &gl_config)) {
        gl_config = buildDefaultConfig();
        if (!writeConfigToFile(NEW_CONFIG_PATH, configToRaw(gl_config))) {
            printf("Failed to write config to file\n");
            reset();
        }
        printf("Default config generated\n");
    }
    else
        printf("Config loaded from file\n");


    // Try and get an IP
    printf("Attempting to use ethernet...\r\n");
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
            printf("We didn't get an IP via DHCP. Continuing wihtout internet.\r\n");
            eth_succ = false;
        }
    }


    //Read the model and checks version.
    vector<uint8_t> model_raw;  //entire model is stored in this vector
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