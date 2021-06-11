#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

python load_lm.py /srv/scratch6/kew/lm_data/rrgen_de/rev_resp.sp-lm-data-bin --chkpt /srv/scratch6/kew/lm_data/rrgen_de/lm_cond/checkpoint_best.pt

"""

import torch
# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel

# List available models
# torch.hub.list('pytorch/fairseq')  # [..., 'transformer_lm.wmt19.en', ...]

breakpoint()

# Load an English LM trained on WMT'19 News Crawl data
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')
en_lm.eval()  # disable dropout

# Move model to GPU
en_lm.cuda()

# Sample from the language model
en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
# "Barack Obama is coming to Sydney and New Zealand (...)"

# Compute perplexity for a sequence
en_lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp()
# tensor(15.1474)

