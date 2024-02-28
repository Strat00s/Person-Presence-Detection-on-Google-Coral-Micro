#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
./build.sh my_project

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

if [[ $1 == reflash ]]; then
    ./flash.sh my_project
else
    ./flash.sh my_project --nodata
fi

if [ $? -ne 0 ]; then
    echo "Flash failed"
    exit 2
fi

./monitor.sh