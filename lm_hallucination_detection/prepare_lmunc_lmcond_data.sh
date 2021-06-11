#!/usr/bin/env bash
# -*- coding: utf-8 -*-


SCRATCH=/srv/scratch6/kew/lm_data/rrgen_de

for suffix in response.sp rev_resp.sp
do
    PREP_DATA=$SCRATCH/$suffix-lm-data-bin/
    echo "prepare $suffix in $PREP_DATA"
    fairseq-preprocess \
        --only-source \
        --trainpref $SCRATCH/raw/train.$suffix \
        --validpref $SCRATCH/raw/valid.$suffix \
        --testpref $SCRATCH/raw/test.$suffix \
        --destdir $PREP_DATA/ \
        --dataset-impl mmap \
        --workers 20;
done

echo "done"