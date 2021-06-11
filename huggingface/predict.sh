#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
home=/home/user/kew
transformers=$home/INSTALLS/transformers

workdir=/srv/scratch6/kew/bart_mbart/
finetuned=$workdir/en_sum_response_test/

export CUDA_VISIBLE_DEVICES=$GPU

python $transformers/examples/seq2seq/run_summarization.py \
    --model_name_or_path $finetuned \
    --do_predict \
    --train_file $workdir/data/en/raw_csv_for_huggingface/train.csv \
    --validation_file $workdir/data/en/raw_csv_for_huggingface/valid.csv \
    --test_file $workdir/data/en/raw_csv_for_huggingface/respo2021_01.csv \
    --text_column review \
    --summary_column response \
    --output_dir $finetuned/predictions/2021_01/ \
    --per_device_eval_batch_size=10 \
    --predict_with_generate \
    --preprocessing_num_workers 10 \
    --max_source_length 512 \
    --max_target_length 512 \
    --num_beams 5

