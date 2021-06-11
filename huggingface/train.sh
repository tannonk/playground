#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU=$1
home=/home/user/kew
transformers=$home/INSTALLS/transformers

workdir=/srv/scratch6/kew/bart_mbart/

export CUDA_VISIBLE_DEVICES=$GPU

python $transformers/examples/seq2seq/run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --train_file $workdir/data/en/raw_csv_for_huggingface/train.csv \
    --validation_file $workdir/data/en/raw_csv_for_huggingface/valid.csv \
    --text_column review \
    --summary_column response \
    --output_dir $workdir/en_sum_response_test \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --preprocessing_num_workers 4 \
    --max_source_length 512 \
    --max_target_length 512 \
    --num_beams 5


