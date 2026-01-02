#!/bin/bash

# 1. Setup Paths
EXP_NAME="tennis-vest"
ROOT_DIR="./videos/${EXP_NAME}"
FLOW_DIR="${ROOT_DIR}_flow"
RAFT_MODEL_PATH="./data_preprocessing/RAFT/models/raft-sintel.pth"
SEGMENT_INFO_PATH="./videos/${EXP_NAME}/segmentation.json"

# 2. Run RAFT to get flow (Optional if already done)
if [ ! -d "$FLOW_DIR" ]; then
    echo "Generating Optical Flow..."
    python data_preprocessing/RAFT/demo.py \
        --model=$RAFT_MODEL_PATH \
        --path=$ROOT_DIR \
        --outdir=$FLOW_DIR \
        --mixed_precision
fi

# 3. Run Motion Segmentation Preprocessing
echo "Running Adaptive Motion Segmentation..."
python preprocess_segmentation.py \
    --root_dir $ROOT_DIR \
    --save_path $SEGMENT_INFO_PATH \
    --threshold_factor 0.5

# 4. Run CoDeF Training
echo "Starting Training..."
python train.py \
    --root_dir $ROOT_DIR \
    --exp_name $EXP_NAME \
    --model_save_path ./ckpts/$EXP_NAME \
    --config configs/base.yaml 