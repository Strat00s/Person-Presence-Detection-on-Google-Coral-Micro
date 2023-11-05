//draw green line from corner to corner
//for (int i = 0; i < CameraTask::kWidth * 3; i += 3) {
//    buf[i     + i * CameraTask::kWidth] = 0;
//    buf[i + 1 + i * CameraTask::kWidth] = 255;
//    buf[i + 2 + i * CameraTask::kWidth] = 0;
//}

#include <cstdio>
#include <vector>

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


using namespace coralmicro;
using namespace std;


#define INDEX_FILE_NAME "/coral_micro_camera.html"
#define STREAM_PATH     "/camera_stream"
#define LOAD_MASK_PATH  "/load_mask"
#define SAVE_MASK_PATH  "/save_mask"

#define IMG_WIDTH  320
#define IMG_HEIGHT 320


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


//printf("jpeg size: %d\r\n", jpeg.size());
//
////{"image":" 10
////"}         2
//std::vector<uint8_t> response_data;
//
//appendToVector(response_data, "{\"image\":\"", 10);
//appendToVector(response_data, jpeg);
//appendToVector(response_data, "\"}", 2);

uint32_t frame_num = 0;

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
        auto image = captureImage(IMG_WIDTH, IMG_HEIGHT, CameraFormat::kRgb);
        if (image.size() == 0)
            return {};

        std::vector<uint8_t> jpeg;
        JpegCompressRgb(image.data(), IMG_WIDTH, IMG_HEIGHT, 75, &jpeg);
        return jpeg;
    }

    else if (StrEndsWith(uri, LOAD_MASK_PATH)) {
        return {};
    }
    else if (StrEndsWith(uri, SAVE_MASK_PATH)) {
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

        printf("Is masked? %d\r\n", isMasked(0, 50, 0, 50, image_mask.grid));

    }
}


extern "C" void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    printf("Camera testing\r\n");


    //initialize default mask
    image_mask.height = 32;
    image_mask.width = 32;
    image_mask.grid.reserve(image_mask.height);
    for (int i = 0; i < image_mask.height * image_mask.width; i++)
        image_mask.grid.push_back(0);


    CameraTask::GetSingleton()->SetPower(true);
    CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

    std::string usb_ip;
    if (GetUsbIpAddress(&usb_ip))
        printf("Serving on: http://%s\r\n", usb_ip.c_str());
    else {
        printf("Failed to start webserver/ethernet adapter!");
        vTaskSuspend(nullptr);
    }

    PostHttpServer http_server;
    http_server.addPostPath(SAVE_MASK_PATH);
    http_server.registerPostFinishedCallback(onPostFinished);
    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);

    vTaskSuspend(nullptr);
}
