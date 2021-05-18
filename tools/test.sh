#!/bin/bash

CONFIG=$1
GPUS=$2
EPOCH=$3
PORT=${PORT:-29500}

CFG_FILE="./configs/${CONFIG}.py"
PTH_FILE="./work_dirs/${CONFIG}/epoch_${EPOCH}.pth"
OUT_FILE="./${CONFIG}.pkl"

module load eth_proxy
source ../venv/bin/activate

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CFG_FILE $PTH_FILE --eval track --out $OUT_FILE \
    --launcher pytorch ${@:4} 
