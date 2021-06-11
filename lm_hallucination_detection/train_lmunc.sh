#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU=$1
SCRATCH=/srv/scratch6/kew/lm_data
PREP_DATA="$SCRATCH/rrgen_de/response.sp-lm-data-bin"
MODEL=$SCRATCH/rrgen_de/lm_unc

export CUDA_VISIBLE_DEVICES=$GPU

fairseq-train --task language_modeling \
  $PREP_DATA \
  --save-dir $MODEL \
  --arch transformer_lm_gpt2_tiny --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --sample-break-mode complete_doc \
  --max-tokens 1024 --update-freq 32 \
  --max-update 50000 \
  --skip-invalid-size-inputs-valid-test
