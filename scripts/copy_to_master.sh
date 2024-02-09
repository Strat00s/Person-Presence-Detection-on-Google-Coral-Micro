#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
cp -r coralmicro/apps/my_project/* src/
cp -r coralmicro/apps/camera_testing/* testing/camera_testing