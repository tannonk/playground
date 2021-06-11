#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# TO RUN:
# bash -i setup_env.sh

home=/home/user/kew/
fairseq=$home/INSTALLS/pytorch_fairseq
workdir=$home/projects/bart_fairseq

conda create -y --name fairseq_bart python=3.8 pip

#set +euo pipefail
# set +eu
# make conda available to current subshell
source $home/anaconda3/etc/profile.d/conda.sh
conda activate fairseq_bart
# set -eu

pip3 install -r $workdir/requirements.txt

pip install --editable $fairseq

echo ""
echo "done!"
echo ""
echo "run: conda activate fairseq_bart"
echo ""
