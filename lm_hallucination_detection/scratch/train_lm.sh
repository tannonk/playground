#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU=$1
SCRATCH=/srv/scratch6/kew/lm_data
PREP_DATA=$SCRATCH/wikitext-2/data-bin/
MODEL=$PREP_DATA/../checkpoints/transformer_wikitext-2_2

export CUDA_VISIBLE_DEVICES=$GPU

fairseq-train --task language_modeling \
  $PREP_DATA \
  --save-dir $MODEL \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --max-update 5000