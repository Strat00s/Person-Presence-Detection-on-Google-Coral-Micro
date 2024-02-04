#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
cp -r src/* coralmicro/apps/my_project
cp -r testing/camera_testing/* coralmicro/apps/camera_testing
