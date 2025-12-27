#!/bin/bash

# 分布式训练启动脚本
# Usage: ./train_ddp.sh [gpu_ids]

# 设置要使用的GPU ID，默认使用所有可用GPU
GPU_IDS=${1:-"0,2"}  # 可以传入如 "0,2" 或 "0,1,2" 等格式

echo "Starting distributed training on GPUs: $GPU_IDS"

# 将GPU IDs转换为数组，计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Number of GPUs: $NUM_GPUS"

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 启动分布式训练
python train_ddp.py --world_size $NUM_GPUS

echo "Distributed training completed"