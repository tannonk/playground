#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


files = [
    '/srv/scratch6/kew/lm_data/rrgen_de/raw/train.rev_resp',
    '/srv/scratch6/kew/lm_data/rrgen_de/raw/valid.rev_resp',
    ]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(vocab_size=10000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "<BOS>",
    "<EOS>",
    "<endtitle>"
])

# ensures that special tokens aren't split!
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train(files, trainer)

tokenizer.save("/srv/scratch6/kew/lm_data/rrgen_de/huggingface/tokenizer-de_rrgen.json")