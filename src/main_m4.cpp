#include "libs/base/ipc_m4.h"
#include "libs/base/main_freertos_m4.h"
#include "libs/base/mutex.h"
#include "libs/base/led.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "libs/camera/camera.h"

#include "ipc_message.hpp"
#include "structs.hpp"



using namespace coralmicro;
using namespace std;


std::vector<bbox_t> bboxes;
std::vector<imageVector_t> images;

imageVector_t m7_image;
imageVector_t m4_image;

SemaphoreHandle_t bbox_mux;
SemaphoreHandle_t image_mux; //local m4 only mux for image access

TaskHandle_t main_task_handle;
TaskHandle_t camera_task_handle;
TaskHandle_t web_task_handle;

CameraFrameFormat frame;
int image_size = 320;


#define CORE_TAG "[M4] "

void printfM4(const char* format, ...) {
    va_list args;
    va_start(args, format);

    // Calculate the size of the new format string with prefix and original format
    size_t prefixLength = strlen(CORE_TAG);
    size_t formatLength = strlen(format);
    size_t newFormatLength = prefixLength + formatLength;
    char *newFormat = new char[newFormatLength + 1]; // +1 for the null terminator

    // Create the new format string with the prefix
    strcpy(newFormat, CORE_TAG);
    strcat(newFormat, format);

    // Call vprintf with the new format string and the original variadic arguments
    {
        MulticoreMutexLock lock(IpcMsgGate::PRINT);
        vprintf(newFormat, args);
    }

    // Clean up
    va_end(args);
    delete[] newFormat;
}


void HandleM7Message(const uint8_t data[kIpcMessageBufferDataSize]) {
    const auto *msg = reinterpret_cast<const IpcMessageStruct *>(data);
    
    if (msg->type == IpcMsgType::BBOX) {
        {
            MulticoreMutexLock lock(msg->gate);
            for (int i = 0; i < msg->item_cnt * 6; i += 6) {
                bboxes.push_back({
                    .xmin  = (*msg->data)[i],
                    .ymin  = (*msg->data)[i + 1],
                    .xmax  = (*msg->data)[i + 2],
                    .ymax  = (*msg->data)[i + 3],
                    .score = (*msg->data)[i + 4],
                    .type  = (*msg->data)[i + 5]
                });
            }
        }
    }

    if (msg->type == IpcMsgType::IMAGE_SIZE) {
        {
            MulticoreMutexLock lock(msg->gate);
            image_size = (*msg->data)[0];
        }
        xTaskNotifyGive(main_task_handle);
    }
}

static void cameraTask(void *args) {
    while (true) {
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
}

extern "C" [[noreturn]] void app_main(void *param) {
    (void)param;
    printfM4("start\n");

    main_task_handle = xTaskGetCurrentTaskHandle();
    bbox_mux = xSemaphoreCreateMutex();
    image_mux = xSemaphoreCreateMutex();

    IpcM4::GetSingleton()->RegisterAppMessageHandler(HandleM7Message);

    //wait for initial image size from M7
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    printfM4("Got image size from M7: %d\n", image_size);

    //CameraTask::GetSingleton()->Init(I2C5Handle());
    //CameraTask::GetSingleton()->SetPower(true);
    //CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);
    //m4_image.resize(image_size * image_size * 3);
    //frame = CameraFrameFormat{
    //    CameraFormat::kRgb,
    //    CameraFilterMethod::kBilinear,
    //    CameraRotation::k270,
    //    image_size,
    //    image_size,
    //    true,
    //    m4_image.data(),
    //    true
    //};

    while (true) {
        printfM4("ALIVE\n");
        vTaskDelay(1333);
    }

    //IpcM4::GetSingleton()->RegisterAppMessageHandler(HandleM7Message);
    vTaskSuspend(nullptr);
}