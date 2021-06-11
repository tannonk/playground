#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# For reference, see
# https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md

set -e

FAIRSEQ=/home/user/kew/INSTALLS/pytorch_fairseq/
WORKDIR=/srv/scratch6/kew/bart_mbart/
DATA_DIR=$WORKDIR/data/en/
EXP_DIR=$WORKDIR/210422_respo_bart/

## extract from pickled dataframe
mkdir -p $EXP_DIR/data/raw/
mkdir -p $EXP_DIR/data/bpe/

python pickle_to_aligned.py $DATA_DIR/en_respo.pkl $EXP_DIR/data/raw/
python pickle_to_aligned.py $DATA_DIR/respondelligent_2021_01_en.sent_seq.scored.pkl $EXP_DIR/data/raw/

for SPLIT in train valid test respo2021_01
do
  for LANG in review response
  do
    python $FAIRSEQ/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json $WORKDIR/bart_base/encoder.json \
    --vocab-bpe $WORKDIR/bart_base/vocab.bpe \
    --inputs $EXP_DIR/data/raw/$SPLIT.$LANG \
    --outputs $EXP_DIR/data/bpe/$SPLIT.$LANG \
    --workers 60 \
    --keep-empty;
  done
done

echo ""
echo "finished applying BPE"
echo ""

fairseq-preprocess \
  --source-lang review \
  --target-lang response \
  --trainpref "$EXP_DIR/data/bpe/train" \
  --validpref "$EXP_DIR/data/bpe/valid" \
  --testpref "$EXP_DIR/data/bpe/test","$EXP_DIR/data/bpe/respo2021_01" \
  --destdir "$EXP_DIR/data/bin/" \
  --workers 60 \
  --srcdict "$WORKDIR/bart_base/bart.base/dict.txt" \
  --tgtdict "$WORKDIR/bart_base/bart.base/dict.txt";

echo ""
echo "finished preprocessing data"
echo ""
