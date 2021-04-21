#!/bin/bash
CONFIG=$1

bsub -n 6 -W 60 -J debug \
    -R "rusage[mem=5000,ngpus_excl_p=1]" \
    -R "select[gpu_model0==GeForceRTX2080Ti]" \
    "./tools/debug.sh ${CONFIG}"