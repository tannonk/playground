#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

SCRATCH=/srv/scratch6/kew/lm_data/rrgen_de

echo "preparing LM_COND data..."
fairseq-preprocess \
    --only-source \
    --trainpref $SCRATCH/raw/train.rev_resp.sp \
    --validpref $SCRATCH/raw/valid.rev_resp.sp \
    --testpref $SCRATCH/raw/test.rev_resp.sp \
    --destdir $SCRATCH/lm_cond/ \
    --dataset-impl mmap \
    --bpe sentencepiece \
    --workers 20;

cp $SCRATCH/sentencepiece.model $SCRATCH/lm_cond/sentencepiece.bpe.model

echo "preparing LM_UNC data (reusing dictionary from LM_COND)..."
fairseq-preprocess \
    --only-source \
    --srcdict $SCRATCH/lm_cond/dict.txt \
    --trainpref $SCRATCH/raw/train.response.sp \
    --validpref $SCRATCH/raw/valid.response.sp \
    --testpref $SCRATCH/raw/test.response.sp \
    --destdir $SCRATCH/lm_unc/ \
    --dataset-impl mmap \
    --bpe sentencepiece \
    --workers 20;

cp $SCRATCH/sentencepiece.model $SCRATCH/lm_unc/sentencepiece.bpe.model

echo "done"


# for suffix in response rev_resp
# do
#     PREP_DATA=$SCRATCH/$suffix-lm/
#     echo "prepare $suffix in $PREP_DATA"
#     fairseq-preprocess \
#         --only-source \
#         --trainpref $SCRATCH/raw/train.$suffix.sp \
#         --validpref $SCRATCH/raw/valid.$suffix.sp \
#         --testpref $SCRATCH/raw/test.$suffix.sp \
#         --destdir $PREP_DATA/ \
#         --dataset-impl mmap \
#         --bpe sentencepiece \
#         --workers 20;

#     cp $SCRATCH/srv/scratch6/kew/lm_data/rrgen_de/sentencepiece.model $PREP_DATA/sentencepiece.bpe.model
# done
