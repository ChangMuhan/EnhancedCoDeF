#!/bin/bash

EXP_NAME="tennis-vest"
ROOT_DIR="./videos/${EXP_NAME}"
CKPT_DIR="./ckpts/${EXP_NAME}"
FLOW_DIR="${ROOT_DIR}_flow" 
RAFT_MODEL_PATH="./data_preprocessing/RAFT/models/raft-sintel.pth"

if [ ! -d "$FLOW_DIR" ]; then
    echo "Generating Optical Flow..."
    python data_preprocessing/RAFT/demo.py \
        --model=$RAFT_MODEL_PATH \
        --path=$ROOT_DIR \
        --outdir=$FLOW_DIR \
        --mixed_precision
fi

echo "Starting Inference..."
python test.py \
    --root_dir $ROOT_DIR \
    --exp_name $EXP_NAME \
    --weight_path $CKPT_DIR \
    --config configs/base.yaml \
    --gpu 0 \
    --fps 15