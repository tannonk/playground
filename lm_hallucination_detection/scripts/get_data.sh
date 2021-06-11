#!/usr/bin/env bash
# -*- coding: utf-8 -*-


data=/srv/scratch6/kew/lm_data

mkdir -p $data

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -P $data
unzip $data/wikitext-2-v1.zip -d $data

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P $data
unzip $data/wikitext-103-v1.zip -d $data

