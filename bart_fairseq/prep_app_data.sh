#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# For reference, see
# https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md

set -e

FAIRSEQ=/home/user/kew/INSTALLS/pytorch_fairseq/
WORKDIR=/srv/scratch6/kew/bart_mbart/
EXP_DIR=$WORKDIR/210504_rrgen_apps_bart/
DATA_DIR=$EXP_DIR/data/
src="review_pref"
tgt="response"

mkdir -p $DATA_DIR/bpe/
mkdir -p $DATA_DIR/bin/


for SPLIT in train valid test 
do
  for LANG in $src $tgt
  do
    python $FAIRSEQ/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json $WORKDIR/bart_base/encoder.json \
    --vocab-bpe $WORKDIR/bart_base/vocab.bpe \
    --inputs $DATA_DIR/raw/$SPLIT.$LANG \
    --outputs $DATA_DIR/bpe/$SPLIT.$LANG \
    --workers 60 \
    --keep-empty;
  done
done

echo ""
echo "finished applying BPE"
echo ""

fairseq-preprocess \
  --source-lang $src \
  --target-lang $tgt \
  --trainpref "$DATA_DIR/bpe/train" \
  --validpref "$DATA_DIR/bpe/valid" \
  --testpref "$DATA_DIR/bpe/test" \
  --destdir "$DATA_DIR/bin/" \
  --workers 60 \
  --srcdict "$WORKDIR/bart_base/bart.base/dict.txt" \
  --tgtdict "$WORKDIR/bart_base/bart.base/dict.txt";

echo ""
echo "finished preprocessing data"
echo ""
