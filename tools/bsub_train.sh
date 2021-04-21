#!/bin/bash
CONFIG=$1
GPUS=$2
NAME=$3

bsub -n 36 -W 24:00 -J $NAME \
    -R "rusage[mem=5000,ngpus_excl_p=${GPUS}]" \
    -R "select[gpu_model0==GeForceRTX2080Ti]" \
    "./tools/train.sh ${CONFIG} ${GPUS}"