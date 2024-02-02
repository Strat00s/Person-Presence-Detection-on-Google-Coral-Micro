#include "Detector.hh"

#include "libs/libjpeg/jpeg.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/detection.h"

#include "libs/tensorflow/posenet.h"
#include "libs/tensorflow/posenet_decoder_op.h"


Detector::Detector() {

}
Detector::~Detector() {

}


vector<uint8_t> Detector::getImage(uint8_t quality) {
    if (quality > 100)
        quality = 100;

    vector<uint8_t> jpeg;
    //TODO mutex to image
    JpegCompressRgb(this->image.data(), this->width, this->height, quality, &jpeg);
    return jpeg;
}
vector<uint8_t> Detector::getRawImage() const {
    //TODO mutex
    return this->image;
}

//TODO mutex
bool Detector::isDetected() const {
    return this->detected;
}

int Detector::getWidth() const {
    return this->width;
}
int Detector::getHeight() const {
    return this->height;
}

int Detector::start() {
    if (!initialized)
        return -1;

    if (handle == nullptr) {
        xTaskCreate(Detector::mainTask, "detector", 8192, this, configMAX_PRIORITIES - 1, &handle);
        stopped = false;
        return 0;
    }

    if (stopped) {
        vTaskResume(handle);
        stopped = false;
    }

    return 0;
}

void Detector::stop() {
    if (!stopped) {
        vTaskSuspend(handle);
        stopped = true;
    }
}

int Detector::reset() {
    stop();
    vTaskDelete(handle);
    handle = nullptr;
    detected = false;
    stopped = true;
    start();
}


int Detector::init(CameraFrameFormat frame_format, const char *model_path) {
    //create camera frame
    this->frame_format = frame_format;
    this->width = frame_format.width;
    this->height = frame_format.height;

    // expecting this program to run on M7 by default

    // power on the camera
    if (CameraTask::GetSingleton()->SetPower(true))
        return 1;
    
    // set trigger mode
    if (CameraTask::GetSingleton()->Enable(CameraMode::kTrigger))
        return 2;

    // set edgetpu speed
    if (!EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax))
        return 3;

    // load model from memory
    std::vector<uint8_t> model_raw;  //entire model is stored in this vector
    if (!LfsReadFile(model_path, &model_raw))
        return 4;

    // confirm model version
    auto model = tflite::GetModel(model_raw.data());
    if (model->version() != TFLITE_SCHEMA_VERSION)
        return 5;

    
    // TODO variable operator registration
    //create micro interpreter
    resolver.AddDequantize();
    resolver.AddDetectionPostprocess();
    resolver.AddCustom(kCustomOp, RegisterCustomOp());
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_area, TENSOR_ARENA_SIZE, &error_reporter);
    if (interpreter->AllocateTensors() != kTfLiteOk)
        return 6;

    // the model must have only 1 input
    if (interpreter->inputs().size() != 1)
        return 7;

    initialized = true;

    return 0;
}


void Detector::mainTask(void *pvParameters) {
    Detector *_this = static_cast<Detector *>(pvParameters);

    while (true) {
        auto input_tensor = _this->interpreter->input(0);
        auto model_height = input_tensor->dims->data[1];
        auto model_width  = input_tensor->dims->data[2];

        CameraTask::GetSingleton()->Trigger();
        if (!CameraTask::GetSingleton()->GetFrame({_this->frame_format})) {
            _this->image.clear();
        }

        //copy image data to tensor
        std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), _this->image.data(), _this->image.size());

        if (_this->interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\r\n");
            //TODO status and stop
            continue;
        }

        //get 5 results
        std::vector<tensorflow::Object> results = tensorflow::GetDetectionResults(_this->interpreter, 0.5);
        if (results.empty()) {
            printf("Nothing detected (below confidence threshold)\r\n");
            _this->detected = false;
        }
        //printf("Number of objects: %d \r\n", results.size());
        //printf("%s\r\n", tensorflow::FormatDetectionOutput(results).c_str());
        _this->detected = true;
    }
}
