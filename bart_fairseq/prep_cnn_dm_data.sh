#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# For reference, see
# https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md

set -e

FAIRSEQ=/home/user/kew/INSTALLS/ffairseq/
WORKDIR=/home/user/kew/projects/bart_fairseq/
DATA_DIR=/srv/scratch6/kew/bart_mbart/data/cnn_dm/

# For reference, see https://github.com/artmatsak/cnn-dailymail

# wget -N 'https://drive.google.com/u/0/uc?export=download&confirm=M0AH&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs' -P $DATA_DIR
# wget -N 'https://drive.google.com/u/0/uc?export=download&confirm=IkJS&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -P $DATA_DIR

# tar -xzvf $DATA_DIR/cnn_stories.tgz -C $DATA_DIR
# tar -xzvf $DATA_DIR/dailymail_stories.tgz -C $DATA_DIR

# cd $WORKDIR/cnn-dailymail-master/
# python make_datafiles.py $DATA_DIR/cnn/stories/ $DATA_DIR/dailymail/stories/ $DATA_DIR/cnn_dm/
# cd $WORKDIR

# echo ""
# echo "finished making data files"
# echo ""

for SPLIT in train val
do
  for LANG in source target
  do
    python $FAIRSEQ/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json $WORKDIR/bart_base/encoder.json \
    --vocab-bpe $WORKDIR/bart_base/vocab.bpe \
    --inputs "$DATA_DIR/$SPLIT.$LANG" \
    --outputs "$DATA_DIR/bpe/$SPLIT.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

echo ""
echo "finished applying BPE"
echo ""

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref $DATA_DIR/bpe/train \
  --validpref $DATA_DIR/bpe/valid \
  --destdir $DATA_DIR/bin/ \
  --workers 60 \
  --srcdict $WORKDIR/bart_base/dict.txt \
  --tgtdict $WORKDIR/bart_base/dict.txt;

echo ""
echo "finished preprocessing data"
echo ""
