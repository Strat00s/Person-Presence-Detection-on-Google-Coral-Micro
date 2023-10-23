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