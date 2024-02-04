#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR
$SCRIPT_DIR/copy_from_master.sh
$SCRIPT_DIR/build.sh my_project
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi
$SCRIPT_DIR/flash.sh my_project
if [ $? -ne 0 ]; then
    echo "Flash failed"
    exit 2
fi
$SCRIPT_DIR/monitor.sh