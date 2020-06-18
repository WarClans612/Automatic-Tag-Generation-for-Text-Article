#!/bin/bash

cmake -DCMAKE_PREFIX_PATH="${HOME}/Course/Automatic-Tag-Generation-for-Text-Article/libtorch;${HOME}/Course/Automatic-Tag-Generation-for-Text-Article/pybind11" ..
#    -DCUDNN_INCLUDE_PATH="/usr/local/cuda/include" \
#    -DCUDNN_LIBRARY_PATH="/usr/local/cuda/lib64" \
#    ..
