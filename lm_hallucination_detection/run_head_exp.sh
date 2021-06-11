#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
SCRATCH=/srv/scratch6/kew/lm_data/rrgen_de/head50
suffix=response.sp

PREP_DATA=$SCRATCH/data-bin/

if [[ ! -f $PREP_DATA/dict.txt ]]; then
    echo "prepare $suffix in $PREP_DATA"
    fairseq-preprocess \
        --only-source --task language_modeling \
        --trainpref $SCRATCH/train.$suffix \
        --validpref $SCRATCH/valid.$suffix \
        --testpref $SCRATCH/test.$suffix \
        --destdir $PREP_DATA/ \
        --dataset-impl mmap \
        --workers 5;
    echo "done prep data"
else
    echo "using data prepared earlier..."
fi

export CUDA_VISIBLE_DEVICES=$GPU

fairseq-train --task language_modeling \
    $PREP_DATA \
    --dataset-impl mmap \
    --arch transformer_lm_gpt2_tiny --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --sample-break-mode complete_doc \
    --max-tokens 1024 --update-freq 2 \
    --max-update 500 --no-save --skip-invalid-size-inputs-valid-test