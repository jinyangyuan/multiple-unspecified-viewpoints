#!/bin/bash

function run_model {
    export CUDA_VISIBLE_DEVICES=${gpu_train_list}
    num_gpus=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))
    port=20000
    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:${port} \
        --nnodes=1 \
        --nproc-per-node=${num_gpus} \
        ${run_file} \
        --config-path ${config_path} \
        --config-name ${config_name_prefix}_single \
        data_name=${data_name}
    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:${port} \
        --nnodes=1 \
        --nproc-per-node=${num_gpus} \
        ${run_file} \
        --config-path ${config_path} \
        --config-name ${config_name_prefix}_multi \
        data_name=${data_name} \
        predicting=false
    export CUDA_VISIBLE_DEVICES=${gpu_test}
    python \
        ${run_file} \
        --config-path ${config_path} \
        --config-name ${config_name_prefix}_multi \
        data_name=${data_name} \
        training=false
}

run_file=../src/main.py
config_path=../exp_multi

data_name=clevr
config_name_prefix=config_blender
gpu_train_list='0'
gpu_test='0'
run_model

data_name=shop
config_name_prefix=config_blender
gpu_train_list='0'
gpu_test='0'
run_model

# data_name=gso
# config_name_prefix=config_kubric
# gpu_train_list='0'
# gpu_test='0'
# run_model

# data_name=shapenet
# config_name_prefix=config_kubric
# gpu_train_list='0'
# gpu_test='0'
# run_model
