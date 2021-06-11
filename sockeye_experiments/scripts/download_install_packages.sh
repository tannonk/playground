#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

tools=$base/tools
mkdir -p $tools

echo "Make sure this script is executed AFTER you have activated a virtualenv"

# CUDA version on instance
CUDA_VERSION=102

# install Sockeye

## Method A: install from PyPi
wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements/requirements.gpu-cu${CUDA_VERSION}.txt
pip install sockeye --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
rm requirements.gpu-cu${CUDA_VERSION}.txt

# install BPE/sentencepiece library
pip install sentencepiece

# install tensorboard for monitoring training progress
pip install tensorboard

pip install mxboard
