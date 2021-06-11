#!/usr/bin/env bash
# -*- coding: utf-8 -*-

GPU=$1
# FAIRSEQ=/home/user/kew/INSTALLS/pytorch_fairseq/
WORKDIR=/srv/scratch6/kew/bart_mbart/
BART_PATH=$WORKDIR/bart_base/bart.base/model.pt
DATA_DIR=$WORKDIR/210504_rrgen_apps_bart/data/bin/
OUTDIR=$WORKDIR/210504_rrgen_apps_bart/checkpoints/
src="review_pref"
tgt="response"

# DATA_DIR=$WORKDIR/data/cnn_dm/
# OUTDIR=$WORKDIR/bart_base/cnn_finetune/checkoints

# hyperparams
TOTAL_NUM_UPDATES=20000
NUM_EPOCHS=8
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048 # maximum number of tokens in a batch
UPDATE_FREQ=2

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU 

fairseq-train $DATA_DIR \
    --restore-file $BART_PATH \
    --task translation \
    --arch bart_base \
    --source-lang $src --target-lang $tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ --required-batch-size-multiple 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch $NUM_EPOCHS \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $OUTDIR \
    --find-unused-parameters;

