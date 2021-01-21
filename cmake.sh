#!/bin/bash

# copy to build directory and run ./cmake.sh

INSTALL_DIR=$(pwd)/../installdir
USER_SOURCE=circuits/qft.c
OUTPUT_EXE=qft
DISTRIBUTED=1
MULTITHREADED=1
GPUACCELERATED=0

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DUSER_SOURCE=${USER_SOURCE} \
  -DOUTPUT_EXE=${OUTPUT_EXE} \
  -DDISTRIBUTED=${DISTRIBUTED} \
  -DMULTITHREADED=${MULTITHREADED} \
  -DGPUACCELERATED=${GPUACCELERATED} \
  ..
