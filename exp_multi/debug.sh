#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

run_file=../src/main.py

python ${run_file}
