#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
SCRATCH=/srv/scratch6/kew/lm_data/rrgen_de
MODEL=$SCRATCH/gpt2small/210628
PREP_DATA=$SCRATCH/prep_data_cond/

export CUDA_VISIBLE_DEVICES=$GPU

if [[ ! -d $PREP_DATA ]]; then
  echo "preparing LM_COND data..."
  fairseq-preprocess \
      --only-source \
      --trainpref $SCRATCH/raw/train.rev_resp.sp \
      --validpref $SCRATCH/raw/valid.rev_resp.sp \
      --testpref $SCRATCH/raw/test.rev_resp.sp \
      --destdir $PREP_DATA \
      --dataset-impl mmap \
      --bpe sentencepiece \
      --workers 20;
  echo "finished preparing data"
else
  echo "reusing prepared data in $PREP_DATA ..."
fi

echo ""
echo "beginning training..."

fairseq-train --task language_modeling \
  $PREP_DATA \
  --save-dir $MODEL \
  --arch transformer_lm_gpt2_small --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 2000 --warmup-init-lr 1e-07 \
  --sample-break-mode complete_doc --shorten-method random_crop --tokens-per-sample 768 \
  --max-tokens 768 --update-freq 18 \
  --max-update 25000 \
  --validate-interval-updates 500 --patience 5 \
  --save-interval-updates 1000 \
  --dataset-impl mmap;
  
echo "training complete!"
