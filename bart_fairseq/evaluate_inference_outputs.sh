#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# use environment fair38!!!
# conda activate fair38

set -e

GPU=$1

SCRIPTS=/home/user/kew/fairseq_add_ons/evaluations/
EXP_DIR=/srv/scratch6/kew/bart_mbart/210422_respo_bart/

export CUDA_VISIBLE_DEVICES=$GPU

for ckpnt in 16 20; do
    INF_DIR=$EXP_DIR/inference/checkpoint$ckpnt/
    for file_stem in test respo2021_01; do
        echo "running evalutaion for $INF_DIR/$file_stem.hypo..."
        sed -i 's/^GREETING>/<GREETING>/g' $INF_DIR/$file_stem.hypo

        python $SCRIPTS/evaluate_sockeye_generation_output.py $INF_DIR/$file_stem.hypo \
        --src_file $EXP_DIR/data/raw/$file_stem.review \
        --ref_file $EXP_DIR/data/raw/$file_stem.response \
        --domain_ref $EXP_DIR/data/raw/$file_stem.domain \
        --rating_ref $EXP_DIR/data/raw/$file_stem.rating \
        --source_ref $EXP_DIR/data/raw/$file_stem.source;
    done
done