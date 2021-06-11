#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#! pip install tokenizers
"""
python train_tokenizer.py 

"""

from tokenizers import ByteLevelBPETokenizer

files = [
    '/srv/scratch6/kew/lm_data/rrgen_de/raw/train.rev_resp',
    '/srv/scratch6/kew/lm_data/rrgen_de/raw/valid.rev_resp',
    ]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=files, vocab_size=10000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "<BOS>",
    "<EOS>",
    "<endtitle>",
])

# Save files to disk
tokenizer.save_model('/srv/scratch6/kew/lm_data/rrgen_de/huggingface', "rrgen_de")