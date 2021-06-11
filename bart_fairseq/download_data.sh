#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# home=/home/user/kew/
# workdir=$home/projects/bart_fairseq/
workdir=/srv/scratch6/kew/bart_mbart/bart_base

mkdir -p $workdir

# Download bart.base model
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz -P $workdir
tar -xzvf $workdir/bart.base.tar.gz -C $workdir

wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json -P "$workdir/bart.base" 
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe -P "$workdir/bart.base"
# wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt -P $workdir


echo ""
echo "done!"
echo ""