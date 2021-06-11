#!/usr/bin/env bash
# -*- coding: utf-8 -*-

base=$1 # specify absolute path as first cmd line arg
GPU=$2 # specify as second cmd line arg

if [[ ! -d $base ]]; then
    mkdir -p "$base"
fi

cp requirements.txt "$base"

cd "$base"

# setup venv
virtualenv -p python3 venvs/fairseq

source venvs/fairseq/bin/activate

pip3 install -r requirements.txt
# CUDA version on instance
# CUDA_VERSION=102

# download data
wget https://files.ifi.uzh.ch/cl/archiv/2020/readvisor/mini-rrgen-en.tar.gz
tar -xzvf mini-rrgen-en.tar.gz

# # download fairseq current master branch
# wget https://github.com/pytorch/fairseq/archive/master.zip
# unzip master.zip
# pip install --editable fairseq-master/.

# download fairseq development branch for experiments
wget https://github.com/tannonk/fairseq/archive/readvisor.zip
unzip readvisor.zip # unzip repo
rm readvisor.zip # clean up

pip install --editable fairseq-readvisor/
# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# launch training
CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
mini-rrgen-en/prep/ \
-s review -t response \
--arch rrgen_lstm_arch \
--task rrgen_translation \
--truncate-source \
--user-dir fairseq-readvisor/examples/rrgen/ \
--dataset-impl raw \
--max-epoch 20 \
--max-tokens 10240 --update-freq 1 \
--lr 0.001 --optimizer adam --clip-norm 0.1 \
--encoder-embed-path mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
--decoder-embed-path mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
--encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
--share-all-embeddings \
--encoder-hidden-size 200 \
--decoder-hidden-size 200 \
--use-sentiment alpha_sentiment --use-category domain --use-rating rating --use-length review_length \
--no-save

# # train standard LSTM
# CUDA_VISIBLE_DEVICES=4 fairseq-train \
# mini-rrgen-en/prep \
# -s review -t response \
# --arch lstm \
# --task translation \
# --truncate-source \
# --dataset-impl raw \
# --max-epoch 5 \
# --max-tokens 10240 \
# --lr 0.001 --optimizer adam --clip-norm 0.1 \
# --encoder-embed-path mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
# --decoder-embed-path mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
# --encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
# --share-all-embeddings --encoder-hidden-size 200 --decoder-hidden-size 200 \
# --no-save

# CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
# mini-rrgen-en/prep/ \
# -s review -t response \
# --arch lstm \
# --task translation \
# --truncate-source \
# --user-dir fairseq-readvisor/examples/rrgen/ \
# --dataset-impl raw \
# --max-epoch 20 \
# --max-tokens 10240 --update-freq 1 \
# --lr 0.001 --optimizer adam --clip-norm 0.1 \
# --encoder-embed-path mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
# --decoder-embed-path mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
# --encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
# --share-all-embeddings \
# --encoder-hidden-size 200 \
# --decoder-hidden-size 200 \
# --no-save

# launch training
# CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
# /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/SG0.6_SL3_LR1.8_UP3_LT2/prep \
# -s review -t response \
# --arch rrgen_lstm_arch \
# --task rrgen_translation \
# --truncate-source --truncate-target \
# --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
# --dataset-impl raw \
# --max-epoch 20 \
# --max-tokens 10240 --update-freq 1 \
# --lr 0.001 --optimizer adam --clip-norm 0.1 \
# --encoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
# --decoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
# --encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
# --share-all-embeddings \
# --encoder-hidden-size 200 \
# --decoder-hidden-size 200 \
# --use-sentiment alpha_sentiment --use-category domain --use-rating rating --use-length review_length \
# --no-save
	

# CUDA_VISIBLE_DEVICES=1 fairseq-train \
# $base/mini-rrgen-en/prep/ \
# -s review -t response \
# --arch rrgen_lstm_arch \
# --task rrgen_translation \
# --truncate-source \
# --user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
# --dataset-impl raw \
# --max-epoch 10 \
# --max-tokens 10240 --update-freq 1 \
# --validate-interval-updates 40 \
# --lr 0.001 --optimizer adam --clip-norm 0.1 \
# --encoder-embed-path $base/mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
# --decoder-embed-path $base/mini-rrgen-en/embeddings/en.wiki.bpe.vs10000.d200.w2v.txt \
# --encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
# --share-all-embeddings \
# --encoder-hidden-size 200 \
# --decoder-hidden-size 200 \
# --no-save
