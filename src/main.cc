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
#include "third_party/mjson/src/mjson.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

using namespace coralmicro;
using namespace std;


#define INDEX_FILE_NAME "/coral_micro_camera.html"
#define STREAM_PREFIX "/camera_stream"

#define MODEL_PATH "/models/bodypix_mobilenet_v1_075_324_324_16_quant_decoder_edgetpu.tflite"

#define TENSOR_ARENA_SIZE     2 * 1024 * 1024
#define CONFIDENCE_THRESHOLD  0.5f
#define LOW_CONFIDENCE_RESULT -2


STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, TENSOR_ARENA_SIZE);

tflite::MicroMutableOpResolver<2> resolver;
tflite::MicroInterpreter *interpreter;


HttpServer::Content UriHandler(const char* uri) {
    if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "coral_micro_camera.html"))
        return std::string(INDEX_FILE_NAME);
    
    else if (StrEndsWith(uri, STREAM_PREFIX)) {
        //auto* interpreter   = static_cast<tflite::MicroInterpreter*>(r->ctx->response_cb_data);
        auto* bodypix_input = interpreter->input(0);
        auto model_height   = bodypix_input->dims->data[1];
        auto model_width    = bodypix_input->dims->data[2];

        std::vector<uint8_t> image(model_width * model_height * CameraFormatBpp(CameraFormat::kRgb));
        std::vector<tensorflow::Pose> results;

        auto fmt = CameraFrameFormat{
            CameraFormat::kRgb,
            CameraFilterMethod::kBilinear,
            CameraRotation::k270,
            model_width,
            model_height,
            false,                  //preserver ratio
            image.data(),
            true                    //white balance
        };

        if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
            printf("Failed to get image from camera.\r\n");
            return {};
        }

        std::memcpy(tflite::GetTensorData<uint8_t>(bodypix_input), image.data(), image.size());

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\r\n");
            return {};
        }

        results = tensorflow::GetPosenetOutput(interpreter, CONFIDENCE_THRESHOLD);
        //if (results.empty()) {
        //    printf("Below confidence threshold\r\n");
        //    return {};
        //}

        //const auto& float_segments_tensor = interpreter->output_tensor(5);
        //const auto& float_segments = tflite::GetTensorData<uint8_t>(float_segments_tensor);
        //const auto segment_size = tensorflow::TensorSize(float_segments_tensor);

        TfLiteTensor *output_tensor = interpreter->output(5);
        int segment_size = tensorflow::TensorSize(output_tensor);

        //printf("interpreter output size: %d\r\n", interpreter->outputs_size());
        printf("segment size: %d \r\n", segment_size);
        printf("Success\r\n");

        uint8_t *data = tflite::GetTensorData<uint8_t>(output_tensor);


        std::vector<uint8_t> heatmap(segment_size * 3 * 100);

        int h = 21;
        int w = 21;
        int W = 210;
        int H = 210;
        
        // Calculate the mapping ratio
        float width_ratio = static_cast<float>(W) / w;
        float height_ratio = static_cast<float>(H) / h;
        
        // Loop through heatmap
        for(int y_heat = 0; y_heat < h; ++y_heat) {
            for(int x_heat = 0; x_heat < w; ++x_heat) {
                int heatmap_idx = y_heat * w + x_heat;
                unsigned char heat_value = data[heatmap_idx];

                // Map to image coordinates
                int x_img_start = static_cast<int>(x_heat * width_ratio);
                int y_img_start = static_cast<int>(y_heat * height_ratio);
                int x_img_end = static_cast<int>((x_heat + 1) * width_ratio);
                int y_img_end = static_cast<int>((y_heat + 1) * height_ratio);

                // Modify image pixels
                for(int y_img = y_img_start; y_img < y_img_end; ++y_img) {
                    for(int x_img = x_img_start; x_img < x_img_end; ++x_img) {
                        int image_idx = (y_img * W + x_img) * 3;

                        // Modify RGB data here
                        // For example, adding heat_value to the red channel:
                        heatmap[image_idx] = heat_value;

                        // Repeat for other channels if necessary
                    }
                }
            }
        }

        std::vector<uint8_t> jpeg;
        JpegCompressRgb(heatmap.data(), 210, 210, /*quality=*/100, &jpeg);
        // [end-snippet:jpeg]
        return jpeg;
    }
    return {};

    //jsonrpc_return_success(r, "{%Q: %d, %Q: %d, %Q: %V, %Q: %V}", "width",
    //                     model_width, "height", model_height, "base64_data",
    //                     image.size(), image.data(), "output_mask1",
    //                     segment_size, float_segments);
}


extern "C" void app_main(void* param) {
    (void)param;

    LedSet(Led::kStatus, true);
    printf("Bodypix HTTP test\r\n");

    tflite::MicroErrorReporter error_reporter;
    //Turn on the TPU and get it's context.
    auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
    if (!tpu_context) {
        TF_LITE_REPORT_ERROR(&error_reporter, "ERROR: Failed to get EdgeTpu context!");
        vTaskSuspend(nullptr);
    }

    //Read the model and checks version.
    std::vector<uint8_t> bodypix_tflite_raw;  //entire model is stored in this vector
    if (!LfsReadFile(MODEL_PATH, &bodypix_tflite_raw)) {
        TF_LITE_REPORT_ERROR(&error_reporter, "Failed to load model!");
        vTaskSuspend(nullptr);
    }
    auto* model = tflite::GetModel(bodypix_tflite_raw.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(&error_reporter, "Model schema version is %d, supported is %d", model->version(), TFLITE_SCHEMA_VERSION);
        vTaskSuspend(nullptr);
    }

    //Create a micro interpreter.
    //tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());
    resolver.AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &error_reporter);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&error_reporter, "AllocateTensors failed.");
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
