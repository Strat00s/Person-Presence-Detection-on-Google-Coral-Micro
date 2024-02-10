#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/..
git clone --recurse-submodules -j8 https://github.com/google-coral/coralmicro

# copy the source
cp -r src coralmicro/apps/my_project

# add cmake target
echo "add_subdirectory(my_project)" >> coralmicro/apps/CMakeLists.txt

# set working hidapi version
sed -i 's/^hidapi==.*$/hidapi==0.11.2/' coralmicro/scripts/requirements.txt

# run the setup
setup.sh