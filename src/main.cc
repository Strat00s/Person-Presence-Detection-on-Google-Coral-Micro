/** YOLO tensor data extraction
auto tensor = interpreter->output_tensor(0);
//printf("Outputs size: %d\n", interpreter->outputs_size());
//printf("Dimensions: %d\n", tensor->dims->size);
//printf("size: %d %d %d\n", tensor->dims->data[0], tensor->dims->data[1], tensor->dims->data[2]);
//printf("type: %d\n", tensor->type);
int box_cnt = tensor->dims->data[1];
int info_cnt = tensor->dims->data[2];
auto tensor_data = tflite::GetTensorData<uint8_t>(tensor);

//std::vector<tensorflow::Object> results;
results.clear();
float scale = 0.005149809643626213f;
int zero_point = 4;
for (int i = 0; i < box_cnt; i++) {
    float score = (tensor_data[i * info_cnt + 4] - zero_point) * scale;
    if (score < 0.5)
        continue;
    score = (tensor_data[i * info_cnt + 5] - zero_point) * scale;
    float max_score = (*std::max_element(tensor_data + i * info_cnt + 5, tensor_data + i * info_cnt + 5 + 80) - zero_point) * scale;
    if (score < max_score)
        continue;
    int id = 0;
    printf("%d %d %d %d\n", tensor_data[i * info_cnt], tensor_data[i * info_cnt + 1], tensor_data[i * info_cnt+ 2], tensor_data[i * info_cnt+3]);
    float bx = (tensor_data[i * info_cnt] - zero_point) * scale;
    float by = (tensor_data[i * info_cnt + 1] - zero_point) * scale;
    float bw = (tensor_data[i * info_cnt + 2]) * scale; // width and height should not have zero_point subtracted
    float bh = (tensor_data[i * info_cnt + 3]) * scale; // before the division by 2

    // Convert from center coordinates to top-left and bottom-right coordinates
    float x1 = (bx - bw / 2.0f);
    float y1 = (by - bh / 2.0f);
    float x2 = (bx + bw / 2.0f);
    float y2 = (by + bh / 2.0f);
    results.push_back(
        tensorflow::Object{
            .id = id,
            .score = score,
            .bbox = {
                .ymin = y1,
                .xmin = x1,
                .ymax = y2,
                .xmax = x2
            }
        }
    );
}
*/

#include <cstdio>
#include <vector>

#include "libs/base/timer.h"
#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "libs/base/filesystem.h"
//#include "libs/rpc/rpc_http_server.h"
#include "libs/tensorflow/posenet.h"
#include "libs/tensorflow/posenet_decoder_op.h"
#include "libs/tpu/edgetpu_manager.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "libs/base/gpio.h"
#include "libs/tensorflow/detection.h"
#include "libs/tpu/edgetpu_op.h"

#include "libs/base/ethernet.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/dns.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/tcpip.h"
#include "libs/base/check.h"
//#include "x86_64-linux-gnu/sys/time.h"

#include "local_libs/http_server.h"

using namespace coralmicro;
using namespace std;


#define INDEX_FILE_NAME "/coral_micro_camera.html"
#define STREAM_PATH     "/camera_stream"
#define LOAD_MASK_PATH  "/load_mask"
#define SAVE_MASK_PATH  "/save_mask"

#define MODEL_PATH "/models/my_models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"


#define TENSOR_ARENA_SIZE     8 * 1024 * 1024
#define CONFIDENCE_THRESHOLD  0.5f
#define LOW_CONFIDENCE_RESULT -2

//#define IMG_WIDTH  320
//#define IMG_HEIGHT 320


STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);


struct mask_struct {
    int width;
    int height;
    std::vector<uint8_t> grid;
};

mask_struct image_mask;



void appendToVector(std::vector<uint8_t> &dst, const char *str, size_t length) {
    dst.reserve(dst.size() + length);
    for (size_t i = 0; i < length; i++) {
        dst.push_back(str[i]);
    }
}

void appendToVector(std::vector<uint8_t> &dst, std::vector<uint8_t> &src) {
    printf("dst size: %d src size: %d\r\n", dst.size(), src.size());
    dst.reserve(dst.size() + src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst.push_back(src[i]);
    }
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


//bool isBoxHalfCovered(const std::vector<std::vector<uint8_t>> &mask, int y_min, int y_max, int x_min, int x_max, int gridSize) {
//    int gridCellSize = 320 / gridSize; // Assuming a 320x320 grid size
//    int totalCells = 0;
//    int nonToggledCells = 0;
//
//    // Iterate through cells within the bounding box
//    for (int i = 0; i < )
//    for (int i = ; i <= endX && i < gridSize; ++i) {
//        for (int j = startY; j <= endY && j < gridSize; ++j) {
//            totalCells++;
//            if (mask[i][j] == 0) { // 0 represents non-toggled in this context
//                nonToggledCells++;
//            }
//        }
//    }
//
//    // Check if at least half of the cells are non-toggled
//    return nonToggledCells >= totalCells / 2;
//}


/** @brief Convert numerical characters in a vector to single integer
 * 
 * @param data vector containing the integer 
 * @param start starting index for converion
 * @param end convert int up to this index (not included)
 * @return 
 */
int numericalVectorToInt(std::vector<uint8_t> data, size_t start, size_t end) {
    printf("start end %d %d\r\n", start, end);
    
    if (end - start > INT_MAX)
        return 0;
    
    int result = 0;
    int mul = 1;
    for (size_t i = 0; i < end - start; i++) {
        char c = data[end - i - 1];
        printf("%c %u\r\n", c, c - '0');
        result += (c - '0') * mul;
        mul *= 10;
    }

    return result;
}


// Helper function to check if the coordinates are inside the box.
bool isInsideBox(int x, int y, int x_min, int x_max, int y_min, int y_max) {
    return (x >= x_min && x <= x_max && y >= y_min && y <= y_max);
}

bool isMasked(int x_min, int x_max, int y_min, int y_max, const std::vector<uint8_t>& grid) {
    
    // Scale factor from the image size to grid size.
    const int scale = 320 / 32;

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
                // Index in the grid (assuming row-major order).
                int index = gridY * 32 + gridX;

                // Check if the cell is non-toggled (assuming 0 is non-toggled).
                if (grid[index] == 0) {
                    coveredCells++;
                }

                totalCells++;
            }
        }
    }

    // Check if at least half of the box is covered by non-toggled cells.
    return coveredCells < totalCells / 2;
}


tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter;
tflite::MicroErrorReporter error_reporter;

std::vector<uint8_t> image;
CameraFrameFormat frame;
int width;
int height;

SemaphoreHandle_t det_mutex;
SemaphoreHandle_t img_mutex;
SemaphoreHandle_t res_mutex;
int status;

TaskHandle_t detect_task_handle;

std::vector<tensorflow::Object> results;

//takes about 106-108ms
static void detectTask(void *args) {
    TickType_t freq = 50;
    TickType_t prev = xTaskGetTickCount();
    status = 0;

    while (true) {
        //vTaskDelayUntil(&prev, freq);
        int ticks = xTaskGetTickCount();

        // get frame
        int ret;
        if (xSemaphoreTake(img_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            ret = CameraTask::GetSingleton()->GetFrame({frame});
            xSemaphoreGive(img_mutex);
        }
        else {
            printf("failed to acquire img mutex\n");
            continue;
        }

        if (!ret) {
            printf("Failed to get image from camera.\r\n");
            if (xSemaphoreTake(det_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                status = -1;
                xSemaphoreGive(det_mutex);
            }
            else
                printf("failed to acquire det mutex\n");
            continue;
        }

        //copy image data to tensor
        if (xSemaphoreTake(img_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            std::memcpy(tflite::GetTensorData<int8_t>(interpreter->input_tensor(0)), image.data(), image.size());
            xSemaphoreGive(img_mutex);
        }
        else {
            printf("failed to acquire img mutex\n");
            continue;
        }


        // run model
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\r\n");
            if (xSemaphoreTake(det_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                status = -2;
                xSemaphoreGive(det_mutex);
            }
            else
                printf("failed to acquire det mutex\n");
            continue;
        }

        // get results and remove anything that is not a person
        int obj_cnt = 0;
        if (xSemaphoreTake(res_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            results = tensorflow::GetDetectionResults(interpreter, 0.65);
            auto new_end = std::remove_if(results.begin(), results.end(), [](tensorflow::Object item) {
                return item.id != 0;
            });
            results.erase(new_end, results.end());

            //obj_cnt = results.size();
            printf("%d objects\n", obj_cnt);
            xSemaphoreGive(res_mutex);
        }
        else {
            printf("failed to acquire res mutex\n");
            continue;
        }

        //printf("%d person objects\n", obj_cnt);
        if (xSemaphoreTake(det_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            status = obj_cnt;
            xSemaphoreGive(det_mutex);
        }
        else
            printf("failed to acquire det mutex\n");
        
        printf("total: %d\n", xTaskGetTickCount() - ticks);
    }
}

void setup() {
    det_mutex = xSemaphoreCreateMutex();
    img_mutex = xSemaphoreCreateMutex();
    res_mutex = xSemaphoreCreateMutex();

    auto* input_tensor = interpreter->input_tensor(0);
    height  = input_tensor->dims->data[1];
    width   = input_tensor->dims->data[2];

    printf("%d %d\n", width, height);
    
    // create and save image storages
    image.resize(width * height * CameraFormatBpp(CameraFormat::kRgb), 0);

    printf("%d\n", image.capacity());

    // create frames for model and preview image
    frame = CameraFrameFormat{
        CameraFormat::kRgb,
        CameraFilterMethod::kNearestNeighbor,
        CameraRotation::k270,
        width,
        height,
        true,
        image.data(),
        true
    };
}

HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "coral_micro_camera.html"))
        return std::string(INDEX_FILE_NAME);
    
    else if (StrEndsWith(uri, STREAM_PATH)) {
        // get status
        if (xSemaphoreTake(det_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            printf("Status: %d\n", status);
            xSemaphoreGive(det_mutex);
        }
        else
            printf("failed to acquire det mutex\n");

        // get image copy
        std::vector<uint8_t> img_copy;
        if (xSemaphoreTake(img_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            img_copy = image;
            xSemaphoreGive(img_mutex);
        }
        else
            printf("failed to get img mutex\n");

        // get result copy
        std::vector<tensorflow::Object> res_copy;
        if (xSemaphoreTake(res_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            res_copy = results;
            xSemaphoreGive(res_mutex);
        }
        else
            printf("failed to get res mutex\n");


        for (auto result : res_copy) {
            //printf("%f %f %f %f\n", result.bbox.xmin, result.bbox.xmax, result.bbox.ymin, result.bbox.ymax);
            int x_min = result.bbox.xmin * width;
            int x_max = result.bbox.xmax * width;
            int y_min = result.bbox.ymin * height;
            int y_max = result.bbox.ymax * height;

            if (isMasked(x_min, x_max, y_min, y_max, image_mask.grid)) {
                drawRectangle(y_min, y_max, x_min, x_max, &img_copy, width, height, 0, 0, 255);
            }
            else
                drawRectangle(y_min, y_max, x_min, x_max, &img_copy, width, height, 0, 255, 0);
        }

        std::vector<uint8_t> jpeg;
        JpegCompressRgb(img_copy.data(), width, height, 100, &jpeg);
        return jpeg;
    }
    else if (StrEndsWith(uri, LOAD_MASK_PATH)) {
        printf("Load not implemented yet");
        return {};
    }
    else if (StrEndsWith(uri, SAVE_MASK_PATH)) {
        printf("Save not implemented yet");
        return {};
    }

    return {};
}

void onPostFinished(payload_uri_t payload_uri) {
    printf("on post finished\r\n");

    if (StrEndsWith(payload_uri.uri, SAVE_MASK_PATH)) {
        if (payload_uri.payload.size() <= 0) {
            printf("Invalid payload size");
            return;
        }

        //TODO I am currently expecting the payload to be in valid format
        auto i = find(payload_uri.payload.begin(), payload_uri.payload.end(), ',');
        int start = 0;
        int end = i - payload_uri.payload.begin();
        image_mask.width = numericalVectorToInt(payload_uri.payload, start, end);

        start = end + 1;
        i = find(payload_uri.payload.begin() + start, payload_uri.payload.end(), ',');
        end = i - payload_uri.payload.begin();
        image_mask.height = numericalVectorToInt(payload_uri.payload, start, end);

        start = end + 1;

        image_mask.grid.clear();
        image_mask.grid.reserve(image_mask.height * image_mask.width);
        for (int i = 0; i < image_mask.height * image_mask.width; i++)
            image_mask.grid.push_back(payload_uri.payload[start + i] - '0');

        printf("Width: %u\r\n", image_mask.width);
        printf("Heigth: %u\r\n", image_mask.height);
        printf("Data: ");
        for (int i = 0; i < image_mask.grid.size(); i++) {
            printf("%u", image_mask.grid[i]);
        }
        printf("\r\n");
    }
}


extern "C" void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);

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

    //initialize default mask
    image_mask.height = 32;
    image_mask.width = 32;
    image_mask.grid.reserve(image_mask.height);
    for (int i = 0; i < image_mask.height * image_mask.width; i++)
        image_mask.grid.push_back(0);



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

    xTaskCreate(detectTask, "detect", 0x4000, nullptr, 0, &detect_task_handle);

    PostHttpServer http_server;
    http_server.addPostPath(SAVE_MASK_PATH);
    http_server.registerPostFinishedCallback(onPostFinished);
    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);

    vTaskSuspend(nullptr);
}