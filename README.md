# Camera-based person detection on Coral Dev Board Micro
This project focuses on performing person presence detection via a camera on a microcontroller.

It uses an SSDLite MobileDet Tensorflow-lite model for detecting objects in images from the camera. The model was trained on the COCO2017 dataset.

The final device used is an unmodified Coral Dev Board Micro with a POE add-on board for internet access. It contains a TPU (Tensor Processor Unit) for inference acceleration and is thus a great option for this project.

The device can be configured through a simple web UI, which allows one to mask parts of the image where detection is not required/wanted, change the confidence threshold, bounding box overlap threshold, and multiple anti-false-positive measures for more reliable detection.


### Running it on the Coralmicro
1. Get a Linux machine.
2. Download/Clone the repository.
3. Run `init.sh` and follow the instructions (requires root when running coralmicro setup).
4. Run `run_all.sh reflash` to build the program, flash it to the board and monitor its serial output.
