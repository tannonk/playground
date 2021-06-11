#!/usr/bin/env bash
# -*- coding: utf-8 -*-

home=/home/user/kew
transformers=$home/INSTALLS/transformers

conda create -y --name transformers python=3.8 pip

#set +euo pipefail
set +eu
# make conda available to current subshell
source $home/anaconda3/etc/profile.d/conda.sh
conda activate transformers
set -eu

pip3 install torch torchvision torchaudio
pip install --editable $transformers
pip install -r $transformers/examples/seq2seq/requirements.txt

echo ""
echo "done!"
echo ""
echo "run conda activate transformers"
echo ""
