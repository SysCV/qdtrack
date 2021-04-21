#!/bin/bash

function rand() {
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000000))
    echo $(($num%$max+$min))
}

CONFIG=$1
GPUS=$2
PORT=$(rand 6000 12000)

CFG_FILE="./configs/${CONFIG}.py"
PTH_FILE="./work_dirs/${CONFIG}/latest.pth"

module load eth_proxy
source ../venv/bin/activate

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CFG_FILE --launcher pytorch ${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CFG_FILE $PTH_FILE --eval track accuracy \
    --launcher pytorch ${@:4}