#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU=$1
SCRATCH=/srv/scratch6/kew/lm_data
DATA=$SCRATCH/rrgen_de/lm_unc
MODEL=$DATA/210615

export CUDA_VISIBLE_DEVICES=$GPU

fairseq-train --task language_modeling \
  $DATA \
  --save-dir $MODEL \
  --arch transformer_lm_gpt2_tiny --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --sample-break-mode complete_doc \
  --max-tokens 16384 --update-freq 2 \
  --max-update 50000 \
  --validate-interval-updates 2500 \
  --save-interval-updates 10000 \
  --dataset-impl mmap \
  --skip-invalid-size-inputs-valid-test
