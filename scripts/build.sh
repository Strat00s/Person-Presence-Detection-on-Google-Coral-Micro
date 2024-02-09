#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
$SCRIPT_DIR/copy_from_master.sh
make -C coralmicro/build/apps/$1 -j4