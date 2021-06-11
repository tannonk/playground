#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
FAIRSEQ=/home/user/kew/INSTALLS/pytorch_fairseq/
WORKDIR=/srv/scratch6/kew/bart_mbart/210504_rrgen_apps_bart
DATA_DIR=$WORKDIR/data/bin/

export CUDA_VISIBLE_DEVICES=$GPU 
chkpt="_best"
FINETUNED=$WORKDIR/checkpoints/checkpoint$chkpt.pt
OUTDIR=$WORKDIR/inference/checkpoint$chkpt/
mkdir -p $OUTDIR

python $FAIRSEQ/examples/bart/summarize.py \
    --model-dir $DATA_DIR \
    --model-file $FINETUNED \
    --src $DATA_DIR/../raw/test.review_pref \
    --out $OUTDIR/test.hypo;


#for chkpt in 4 8 12 16 20; do
#  echo "running inference with checkpoint: $chkpt"
#
#  FINETUNED=$WORKDIR/checkpoints/checkpoint$chkpt.pt
#  OUTDIR=$WORKDIR/inference/checkpoint$chkpt/
#  mkdir -p $OUTDIR
#
#  python $FAIRSEQ/examples/bart/summarize.py \
#    --model-dir $DATA_DIR \
#    --model-file $FINETUNED \
#    --src $DATA_DIR/../raw/test.review \
#    --out $OUTDIR/test.hypo
#
#  python $FAIRSEQ/examples/bart/summarize.py \
#    --model-dir $DATA_DIR \
#    --model-file $FINETUNED \
#    --src $DATA_DIR/../raw/respo2021_01.review \
#    --out $OUTDIR/respo2021_01.hypo
#done

echo ""
echo "done!"
echo ""
