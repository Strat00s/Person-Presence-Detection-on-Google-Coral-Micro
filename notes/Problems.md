Github Submodules are horrible. Developing like that is almost impossible, since submodules count as their own repos.

Camera is bad, but it should be good enough.

When using the TPU, it is very powerful: 20FPS when capturing, streaming and processing human pose data and drawing skeleton over it.

Light. The camera needs lots of light.

Slow upload. When uploading the finish program, it is really slow.

Upload. The device connects as 3 different devices when operating normally and as multiple different HID devices during upload. Since there is no build and upload script for windows. only virtual machine or bare metal linux can be used for developement.

## Camera
Quality is actually ok. But white balancing happens only on initialization
- TODO FIX

TPU is probably connected through USB -> sending large amount of data leads to USB transmit timeout and total system halt.

USB HTTP Server is lows and misses

270 rotation is camera up, port down.

## Time
Capturing an image takes about 30ms
inference takes by far the longest. And it all depends on the model (obviously). But since I can't see into the custom edge tpu operator, I don't know how complex the mdoels are.

Pretty much anything else (except for drawing on the image) is relatively quick.
NMS adds practically no delay.
My guess is that tracking will also consume minimum cpu time

Drawing is slow (obviously). My guess is that context switching and everything adds about 50ms when displaying a single image.
Since drawing onto the image and displaying the webpage takes the longest, I want to move it to the M4 core and have low FPS for preview, while leaving M7 to run inference with the edge tpu (M4 does not have access to the tpu anyways).

TLDR: optimization is a must. But first.

## Exceptions and crashes
no clue, but you can edit coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake to add arguments to build.

## Multicore
Camera capture on M4 take over half a second (compared to about 30ms on M7...)
M4 is just too slow

## Framerate
Sending raw image gives faster inference (about 15ms)
But sending the raw image takes much longer compared to sending a jpeg
Jpeg gives more frames in preview
Raw image gives faster inference in preview
Both are the same speed when not previewing