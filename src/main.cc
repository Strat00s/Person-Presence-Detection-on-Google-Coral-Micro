//#include <cstdio>
//
//#include "libs/base/led.h"
//#include "third_party/freertos_kernel/include/FreeRTOS.h"
//#include "third_party/freertos_kernel/include/task.h"

// Prints "hello world" in the serial console.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e hello_world

//extern "C" [[noreturn]] void app_main(void *param) {
//    (void)param;
//    
//    uint8_t cnt = 0;
//    while(true) {
//        printf("Hello world %d\r\n", cnt++);
//        
//        LedSet(coralmicro::Led::kStatus, true);
//        vTaskDelay(pdMS_TO_TICKS(500));
//        LedSet(coralmicro::Led::kStatus, false);
//        vTaskDelay(pdMS_TO_TICKS(500));
//    }
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


// Hosts an RPC server on the Dev Board Micro that streams camera images
// to a connected client app.

namespace coralmicro {
  namespace {

    constexpr char kIndexFileName[]         = "/coral_micro_camera.html";
    constexpr char kCameraStreamUrlPrefix[] = "/camera_stream";

    HttpServer::Content UriHandler(const char* uri) {
        if (StrEndsWith(uri, "index.shtml") || StrEndsWith(uri, "coral_micro_camera.html")) {
            return std::string(kIndexFileName);
        }

        else if (StrEndsWith(uri, kCameraStreamUrlPrefix)) {
            // [start-snippet:jpeg]
            std::vector<uint8_t> buf(CameraTask::kWidth * CameraTask::kHeight * CameraFormatBpp(CameraFormat::kRgb));
            auto fmt = CameraFrameFormat{
                CameraFormat::kRgb,
                CameraFilterMethod::kBilinear,
                CameraRotation::k0,
                CameraTask::kWidth,
                CameraTask::kHeight,
                /*preserve_ratio=*/false,
                buf.data(),
                /*while_balance=*/true
            };
            if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
                printf("Unable to get frame from camera\r\n");
                return {};
            }

            std::vector<uint8_t> jpeg;
            JpegCompressRgb(buf.data(), fmt.width, fmt.height, /*quality=*/90, &jpeg);
            // [end-snippet:jpeg]
            return jpeg;
        }
        return {};
    }

    void Main() {
        printf("Camera HTTP Example!\r\n");
        // Turn on Status LED to show the board is on.
        LedSet(Led::kStatus, true);

        CameraTask::GetSingleton()->SetPower(true);
        CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

        std::string usb_ip;
        if (GetUsbIpAddress(&usb_ip)) {
            printf("Serving on: http://%s\r\n", usb_ip.c_str());
        }

        HttpServer http_server;
        http_server.AddUriHandler(UriHandler);
        UseHttpServer(&http_server);

        vTaskSuspend(nullptr);
    }
  }  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    coralmicro::Main();
}
