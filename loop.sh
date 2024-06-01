#!/bin/bash

save_dir_prefix="/data1/lzs/test/gsm"
model_path_prefix="/data1/lzs/test/merged/fine"

for i in {0..9}
do
    save_dir="${save_dir_prefix}${i}"
    model_path="${model_path_prefix}${i}"

    CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com \
    python /home/lsz/LLM-coding-test-Ang/gsm8k_eval.py --save_dir "$save_dir" --model_name_or_path "$model_path"
done