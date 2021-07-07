#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU="0"
SCRATCH=/srv/scratch6/kew/lm_data
PREP_DATA=$SCRATCH/wikitext-2/data-bin/
MODEL=$PREP_DATA/checkpoints/transformer_wikitext-2

fairseq-eval-lm $PREP_DATA \
    --path $MODEL/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400