//draw green line from corner to corner
//for (int i = 0; i < CameraTask::kWidth * 3; i += 3) {
//    buf[i     + i * CameraTask::kWidth] = 0;
//    buf[i + 1 + i * CameraTask::kWidth] = 255;
//    buf[i + 2 + i * CameraTask::kWidth] = 0;
//}

#include <cstdio>
#include <vector>

#include "libs/base/timer.h"
#include "libs/base/http_server.h"
#include "apps/camera_testing/libs/http_server.h"
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


using namespace coralmicro;
using namespace std;


#define INDEX_FILE_NAME "/coral_micro_camera.html"
#define STREAM_PATH     "/camera_stream"
#define LOAD_MASK_PATH  "/load_mask"
#define SAVE_MASK_PATH  "/save_mask"

#define MODEL_PATH "/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"

#define TENSOR_ARENA_SIZE     8 * 1024 * 1024
#define CONFIDENCE_THRESHOLD  0.5f
#define LOW_CONFIDENCE_RESULT -2

#define IMG_WIDTH  320
#define IMG_HEIGHT 320


STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);

tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter;


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


vector<uint8_t> captureImage(int width, int height, CameraFormat format) {
    std::vector<uint8_t> image(width * height * CameraFormatBpp(format));

    //create camera frame
    auto fmt = CameraFrameFormat{
        format,
        CameraFilterMethod::kBilinear,
        CameraRotation::k270,
        width,
        height,
        false,
        image.data()
    };

    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
        printf("Failed to get image from camera.\r\n");
        return vector<uint8_t>();
    }

    //printf("Frame %ld\r\n", frame_num++);

    return image;
}


HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "coral_micro_camera.html"))
        return std::string(INDEX_FILE_NAME);
    
    else if (StrEndsWith(uri, STREAM_PATH)) {
        //get input tensor dimensions
        auto* input_tensor = interpreter->input(0);
        auto model_height  = input_tensor->dims->data[1];
        auto model_width   = input_tensor->dims->data[2];

        auto image = captureImage(model_width, model_height, CameraFormat::kRgb);
        if (image.size() == 0)
            return {};


        //copy image data to tensor
        std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image.data(), image.size());

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\r\n");
            return {};
        }

        //get 5 results
        std::vector<tensorflow::Object> results = tensorflow::GetDetectionResults(interpreter, 0.5);
        if (results.empty()) {
            printf("Nothing detected (below confidence threshold)\r\n");
            return {};
        }
        //printf("Number of objects: %d \r\n", results.size());
        //printf("%s\r\n", tensorflow::FormatDetectionOutput(results).c_str());
        
        //draw bounding box around first person
        //TODO measure distance between current and last box to remove false detections (too fast movement and such)
        for (size_t i = 0; i < results.size(); i++) {
            //search for person in results
            if (results[i].id == 0) {
                int x_min = results[i].bbox.xmin * model_width;
                int x_max = results[i].bbox.xmax * model_width;
                int y_min = results[i].bbox.ymin * model_height;
                int y_max = results[i].bbox.ymax * model_height;

                if (isMasked(x_min, x_max, y_min, y_max, image_mask.grid)) {
                    printf("Masked\r\n");
                    continue;
                }
                drawRectangle(y_min, y_max, x_min, x_max, &image, model_width, model_height);
                printf("Not masked\r\n");
                break;
            }
        }

        std::vector<uint8_t> jpeg;
        JpegCompressRgb(image.data(), model_width, model_height, 75, &jpeg);
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
    TimerInit();
    auto idk = TimerMicros();


    LedSet(Led::kStatus, true);
    printf("Bodypix test\r\n");


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

    // Wait for the clock to be set via NTP.
    //struct timeval tv;
    //int gettimeofday_retries = 10;
    //do {
    //    if (gettimeofday(&tv, nullptr) == 0) {
    //        break;
    //    }
    //    gettimeofday_retries--;
    //    vTaskDelay(pdMS_TO_TICKS(1000));
    //} while (gettimeofday_retries);
    //if (!gettimeofday_retries) {
    //    printf("Clock was never set via NTP.\r\n");
    //    return;
    //}


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


    //std::string usb_ip;
    //if (GetUsbIpAddress(&usb_ip))
    //    printf("Serving on: http://%s\r\n", usb_ip.c_str());
    //else {
    //    printf("Failed to start webserver/ethernet adapter!");
    //    vTaskSuspend(nullptr);
    //}

    PostHttpServer http_server;
    http_server.addPostPath(SAVE_MASK_PATH);
    http_server.registerPostFinishedCallback(onPostFinished);
    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);

    printf("END: %llu\r\n", TimerMicros() - idk);

    vTaskSuspend(nullptr);
}
