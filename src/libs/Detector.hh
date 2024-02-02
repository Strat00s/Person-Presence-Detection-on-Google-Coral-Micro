#include <cstdio>
#include <vector>
#include <map>

#include "FreeRTOS.h"
#include "task.h"

#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/tensorflow/utils.h"
#include "libs/base/filesystem.h"


// tensor input memory
#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE 8 * 1024 * 1024
#endif

using namespace coralmicro;
using namespace std;




class Detector {
private:
    bool detected = false;

    int width;              // image width
    int height;             // image height
    vector<uint8_t> image;  // last captured image
    CameraFrameFormat frame_format;
    
    STATIC_TENSOR_ARENA_IN_SDRAM(tensor_area, TENSOR_ARENA_SIZE);
    tflite::MicroMutableOpResolver<3> resolver;
    tflite::MicroInterpreter *interpreter;
    tflite::MicroErrorReporter error_reporter;
    std::shared_ptr<coralmicro::EdgeTpuContext> tpu_context;
    float threshold = 0.5f; //confidence threshold

    TaskHandle_t handle = nullptr;
    bool stopped = true;
    bool initialized = false;

    /** @brief Main underlying detection task*/
    static void mainTask(void *pvParameters);

public:
    Detector(/* args */);
    ~Detector();

    /** @brief Get last captured image as jpeg
     * 
     * @return vector if bytes representing the jpeg
     */
    vector<uint8_t> getImage(uint8_t quality = 75);

    /** @brief Get raw bytes of last capture image
     * 
     * @return vector of raw bytes representing the image of size width * height * bytes_per_pixel
     */
    vector<uint8_t> getRawImage() const;

    /** @brief get preson detection status
     * 
     * @return 
     */
    bool isDetected() const;

    /** @brief Start the detection task
     * 
     * @return int 
     */
    int start();

    /** @brief Stop the detection task
     * 
     * 
     * @return 
     */
    void stop();

    /** @brief Reset the detection task
     * (stop it, delete it, recreate it, start it)
     * 
     * @return 
     */
    int reset();


    int init(CameraFrameFormat frame_format, const char *model_path);

    int getWidth() const;
    int getHeight() const;

    int status();
};

