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


int image_size = 0;
//updated regularly from M4 camera task
//read sparsly from M7 task
image_vector_t *image_before;
//update by M7 detector task
//read by M4 task for writing bboxes and converting it to jpeg
image_vector_t *image_after;
//updated by M4 from image_after and bboxes
//read by M7 to send to webpage
image_vector_t *jpeg;
//updated by M7 after inference
//read by M4 when drawing bounding boxes
bbox_vector_t *bboxes;


TaskHandle_t main_task_handle; //starts all other tasks
//periodically takes pictures and saves it to image_before
TaskHandle_t camera_task_handle;
//periodically reads image_after, bboxes and writes them to jpeg
TaskHandle_t jpeg_draw_task;


void printfM4(const char* format, ...) {
    va_list args;
    va_start(args, format);
    {
        MulticoreMutexLock lock(IpcMsgGate::PRINT);
        printf("[M4] ");
        vprintf(format, args);
    }
    va_end(args);
}

void notifyM7(IpcMsgType type, uint8_t gate = IpcMsgGate::OTHER, int image_size = 0, image_vector_t *img_b = nullptr, image_vector_t *img_a = nullptr, image_vector_t *jpeg = nullptr, bbox_vector_t *bboxes = nullptr) {
    auto *ipc = IpcM4::GetSingleton();
    IpcMessage msg{};
    msg.type = IpcMessageType::kApp;
    auto *app_msg = reinterpret_cast<IpcMessageStruct *>(&msg.message.data);

    app_msg->type       = type;
    app_msg->image_b    = img_b;
    app_msg->image_a    = img_a;
    app_msg->jpeg       = jpeg;
    app_msg->bboxes     = bboxes;
    app_msg->image_size = image_size;
    app_msg->gate       = gate;

    ipc->SendMessage(msg);
}

void HandleM7Message(const uint8_t data[kIpcMessageBufferDataSize]) {
    const auto *msg = reinterpret_cast<const IpcMessageStruct *>(data);

    if (msg->type == IpcMsgType::CONFIG) {
        image_before = msg->image_b;
        image_after  = msg->image_a;
        jpeg         = msg->jpeg;
        bboxes       = msg->bboxes;
        image_size   = msg->image_size;
        xTaskNotifyGive(main_task_handle);
    }

    if (msg->type == IpcMsgType::ACK)
        xTaskNotifyGive(main_task_handle);
}

static void cameraTask(void *args) {
    int ret;

    CameraTask::GetSingleton()->Init(I2C5Handle());
    ret = CameraTask::GetSingleton()->SetPower(false);
    if (!ret)
        printf("Failed to power off\n");
    vTaskDelay(pdMS_TO_TICKS(100));
    ret = CameraTask::GetSingleton()->SetPower(true);
    if (!ret)
        printf("Failed to power on\n");
    ret = CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);
    if (!ret)
        printf("Failed to enable\n");
    image_vector_t image;
    CameraFrameFormat frame;
    {
        MulticoreMutexLock lock(IpcMsgGate::IMAGE_BEFORE);
        image.resize(image_size * image_size * 3);
        printfM4("before size: %d\n", image.size());
        frame = CameraFrameFormat{
            CameraFormat::kRgb,
            CameraFilterMethod::kBilinear,
            CameraRotation::k270,
            image_size,
            image_size,
            true,
            image.data(),
            true
        };
    }

    while (true) {
        int ticks = xTaskGetTickCount();
        CameraTask::GetSingleton()->GetFrame({frame});
        printfM4("Capture took %dms\n", xTaskGetTickCount() - ticks);

        {//mutex lock
            MulticoreMutexLock lock(IpcMsgGate::IMAGE_BEFORE);
            *image_before = image;
        }
    }
}


extern "C" [[noreturn]] void app_main(void *param) {
    (void)param;
    printfM4("start\n");

    main_task_handle = xTaskGetCurrentTaskHandle();
    
    IpcM4::GetSingleton()->RegisterAppMessageHandler(HandleM7Message);

    //wait for config from M7
    //ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    //printfM4("Got config from M7: %d\n", image_size);
    //notifyM7(IpcMsgType::ACK);

    //xTaskCreate(cameraTask, "camera", 4096, nullptr, 3, &camera_task_handle);


    while (true) {
        printfM4("ALIVE");
        vTaskDelay(1333);
    }

    //IpcM4::GetSingleton()->RegisterAppMessageHandler(HandleM7Message);
    vTaskSuspend(nullptr);
}