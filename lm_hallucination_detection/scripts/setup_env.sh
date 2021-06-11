#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# TO RUN:
# bash -i setup_env.sh

ENVNAME=hallu_lm
home=/home/user/kew/
fairseq=$home/INSTALLS/pytorch_fairseq
# workdir=$home/projects/lm_hallucination_detection/

conda create -y --name $ENVNAME python=3.8 pip

# make conda available to current subshell & activate new env
source $home/anaconda3/etc/profile.d/conda.sh && conda activate $ENVNAME

pip3 install -r ./requirements.txt

pip install --editable $fairseq

echo ""
echo "done!"
echo ""
echo "run: conda activate $ENVNAME"
echo ""
