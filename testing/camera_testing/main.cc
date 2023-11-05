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

    printf("Frame %ld\r\n", frame_num++);

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
}


extern "C" void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    printf("Camera testing\r\n");


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
