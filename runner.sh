#!/bin/bash

set -Eeuo pipefail

echo $(dirname $(readlink -f $0))

SCRIPT_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$(readlink -f "${SCRIPT_DIR}")
echo $(readlink -f "${SCRIPT_DIR}")
export PYTHONPATH=$PYTHONPATH:$(readlink -f "${SCRIPT_DIR}/export")
echo $PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export PYOPENGL_PLATFORM=egl

$SCRIPT_DIR/env/bin/python $@
