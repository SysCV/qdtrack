#!/bin/bash

CONFIG=$1

module load eth_proxy
source ../venv/bin/activate

python ./tools/train.py $CONFIG