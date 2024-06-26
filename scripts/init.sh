#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

# clone the coral micro repo
git clone --recurse-submodules -j8 https://github.com/google-coral/coralmicro

# copy the source
cp -r src coralmicro/apps/my_project

# add cmake target
echo "add_subdirectory(my_project)" >> coralmicro/apps/CMakeLists.txt

# set working hidapi version
sed -i 's/^hidapi==.*$/hidapi==0.11.2/' coralmicro/scripts/requirements.txt

# download model from coral website
mkdir coralmicro/models/my_models
wget -P coralmicro/models/my_models https://raw.githubusercontent.com/google-coral/test_data/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite

# run the coral micro setup (requires root access for apt update)
coralmicro/setup.sh

# run initial build
coralmicro/build.sh