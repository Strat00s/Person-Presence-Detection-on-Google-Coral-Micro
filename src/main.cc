//draw green line from corner to corner
//for (int i = 0; i < CameraTask::kWidth * 3; i += 3) {
//    buf[i     + i * CameraTask::kWidth] = 0;
//    buf[i + 1 + i * CameraTask::kWidth] = 255;
//    buf[i + 2 + i * CameraTask::kWidth] = 0;
//}

#include <cstdio>
#include <vector>

#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#include "libs/base/filesystem.h"
#include "libs/rpc/rpc_http_server.h"
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
#define STREAM_PREFIX "/camera_stream"

#define MODEL_PATH "/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"

#define TENSOR_ARENA_SIZE     8 * 1024 * 1024
#define CONFIDENCE_THRESHOLD  0.5f
#define LOW_CONFIDENCE_RESULT -2


STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);

tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter *interpreter;

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


HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "coral_micro_camera.html"))
        return std::string(INDEX_FILE_NAME);
    
    else if (StrEndsWith(uri, STREAM_PREFIX)) {
        //get input tensor dimensions
        auto* input_tensor = interpreter->input(0);
        auto model_height  = input_tensor->dims->data[1];
        auto model_width   = input_tensor->dims->data[2];

        std::vector<uint8_t> image(model_width * model_height * CameraFormatBpp(CameraFormat::kRgb));
        
        //create camera frame
        auto fmt = CameraFrameFormat{
            CameraFormat::kRgb,
            CameraFilterMethod::kBilinear,
            CameraRotation::k270,
            model_width,
            model_height,
            false,
            image.data()
        };

        //CameraTask::GetSingleton()->Trigger();
        if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
            printf("Failed to get image from camera.\r\n");
            return {};
        }

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
        printf("Number of objects: %d \r\n", results.size());
        //printf("%s\r\n", tensorflow::FormatDetectionOutput(results).c_str());
        
        //draw bounding box around person
        for (size_t i = 0; i < results.size(); i++) {
            //search for person in results
            if (results[i].id == 0) {
                int x_min = results[i].bbox.xmin * model_width;
                int x_max = results[i].bbox.xmax * model_width;
                int y_min = results[i].bbox.ymin * model_height;
                int y_max = results[i].bbox.ymax * model_height;
                drawRectangle(y_min, y_max, x_min, x_max, &image, model_width, model_height);
                break;
            }
        }

        //draw boudning boxes for 5 results
        //int r, g, b;
        //for (int i = 0; i < 5; i++) {
        //    switch (i) {
        //        case 0:  r = 0;   g = 255; b = 0;   break;
        //        case 1:  r = 255; g = 0;   b = 0;   break;
        //        case 2:  r = 0;   g = 0;   b = 255; break;
        //        case 3:  r = 0;   g = 128; b = 127; break;
        //        case 4:  r = 128; g = 127; b = 0;   break;
        //        default: r = 0;   g = 255; b = 0;   break;
        //    }
        //    int x_min = results[i].bbox.xmin * model_width;
        //    int x_max = results[i].bbox.xmax * model_width;
        //    int y_min = results[i].bbox.ymin * model_height;
        //    int y_max = results[i].bbox.ymax * model_height;
        //    drawRectangle(y_min, y_max, x_min, x_max, &image, model_width, model_height, r, g, b);
        //}

        std::vector<uint8_t> jpeg;
        JpegCompressRgb(image.data(), model_width, model_height, /*quality=*/75, &jpeg);
        // [end-snippet:jpeg]
        return jpeg;
    }
    return {};
}


extern "C" void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    printf("Bodypix HTTP test\r\n");


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

    std::string usb_ip;
    if (GetUsbIpAddress(&usb_ip))
        printf("Serving on: http://%s\r\n", usb_ip.c_str());
    else {
        printf("Failed to start webserver/ethernet adapter!");
        vTaskSuspend(nullptr);
    }

    HttpServer http_server;
    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);

    vTaskSuspend(nullptr);
}
