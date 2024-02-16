#include <vector>
#include <cstdarg>

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
std::vector<image_vector_t> images;

image_vector_t m7_image;
image_vector_t m4_image;
int image_size = 0;

SemaphoreHandle_t bbox_mux;
SemaphoreHandle_t image_mux; //local m4 only mux for image access

TaskHandle_t main_task_handle;
TaskHandle_t camera_task_handle;
TaskHandle_t web_task_handle;

CameraFrameFormat frame;


#define CORE_TAG "[M4] "

void printfM4(const char* format, ...) {
    va_list args;
    va_start(args, format);

    // Call vprintf with the new format string and the original variadic arguments
    {
        MulticoreMutexLock lock(IpcMsgGate::PRINT);
        printf("[M4] ");
        vprintf(format, args);
    }

    // Clean up
    va_end(args);
}

void notifyM7(IpcMsgType type, bbox_vector_t *bboxes, image_vector_t *image, int image_size, uint8_t gate) {
    auto *ipc = IpcM4::GetSingleton();
    IpcMessage msg{};
    msg.type = IpcMessageType::kApp;
    auto *app_msg = reinterpret_cast<IpcMessageStruct *>(&msg.message.data);
    app_msg->type       = type;
    app_msg->bboxes     = bboxes;
    app_msg->image      = image;
    app_msg->image_size = image_size;
    app_msg->gate       = gate;

    ipc->SendMessage(msg);
}

void HandleM7Message(const uint8_t data[kIpcMessageBufferDataSize]) {
    const auto *msg = reinterpret_cast<const IpcMessageStruct *>(data);
    
    if (msg->type == IpcMsgType::BBOX) {
        {
            MulticoreMutexLock lock(msg->gate);
            bboxes = (*msg->bboxes);
        }
    }

    if (msg->type == IpcMsgType::IMAGE_SIZE) {
        {
            MulticoreMutexLock lock(msg->gate);
            image_size = msg->image_size;
        }

        xTaskNotifyGive(main_task_handle);
    }

    if (msg->type == IpcMsgType::ACK)
        xTaskNotifyGive(main_task_handle);
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
    notifyM7(IpcMsgType::ACK, nullptr, nullptr, 0, IpcMsgGate::OTHER);


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