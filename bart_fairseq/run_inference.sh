#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# changes for 210421_respo_bart
# WORKDIR=/srv/scratch6/kew/bart_mbart/210421_respo_bart/
# for chkpt in 4 8 _best; do


GPU=$1
FAIRSEQ=/home/user/kew/INSTALLS/pytorch_fairseq/
WORKDIR=/srv/scratch6/kew/bart_mbart/210422_respo_bart/
DATA_DIR=$WORKDIR/data/bin/

# /srv/scratch6/kew/bart_mbart/210421_respo_bart/data/raw/test.review

export CUDA_VISIBLE_DEVICES=$GPU 

for chkpt in 4 8 12 16 20; do
  echo "running inference with checkpoint: $chkpt"

  FINETUNED=$WORKDIR/checkpoints/checkpoint$chkpt.pt
  OUTDIR=$WORKDIR/inference/checkpoint$chkpt/
  mkdir -p $OUTDIR

  python $FAIRSEQ/examples/bart/summarize.py \
    --model-dir $DATA_DIR \
    --model-file $FINETUNED \
    --src $DATA_DIR/../raw/test.review \
    --out $OUTDIR/test.hypo

  python $FAIRSEQ/examples/bart/summarize.py \
    --model-dir $DATA_DIR \
    --model-file $FINETUNED \
    --src $DATA_DIR/../raw/respo2021_01.review \
    --out $OUTDIR/respo2021_01.hypo
done

echo ""
echo "done!"
echo ""
