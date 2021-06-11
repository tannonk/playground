#!/usr/bin/env bash
# -*- coding: utf-8 -*-

SCRATCH=/srv/scratch6/kew/lm_data
DATA=$SCRATCH/wikitext-2
PREP_DATA=$DATA/data-bin/

fairseq-preprocess \
    --only-source \
    --trainpref $DATA/wiki.train.tokens \
    --validpref $DATA/wiki.valid.tokens \
    --testpref $DATA/wiki.test.tokens \
    --destdir $PREP_DATA/ \
    --dataset-impl mmap \
    --workers 20