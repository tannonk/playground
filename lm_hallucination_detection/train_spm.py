#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
"""

import sys
from pathlib import Path
from tqdm import tqdm
import sentencepiece as spm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--train_files', nargs='*', required=True, help='list of files to learn sp model')
ap.add_argument('--model_name', required=True, type=Path, help='E.g. /srv/scratch2/kew/fairseq_materials/ssnn_rewrite/de/data_bpe_lc/')
ap.add_argument('--vocab_size', required=False, default=10000, help='size of bpe vocab symbols')
ap.add_argument('--symbols', required=False, type=Path, help='path to txt file containing special symbols (expects one symbol per line).')
args = ap.parse_args()

# user-defined tokens - not to be split
if args.symbols:
    with open(args.symbols, 'r', encoding='utf8') as sf:
        symbols = list(set([sym.strip() for sym in sf.readlines()]))
else:
    symbols = []

# create output directory
outdir = args.model_name.parent

outdir.mkdir(parents=True, exist_ok=True)

print(f'Training sp model on {args.train_files}...')

spm.SentencePieceTrainer.train(input=args.train_files, model_prefix=args.model_name, vocab_size=args.vocab_size, user_defined_symbols=symbols, model_type='bpe')

print(f'Training complete. See {args.model_name}.model / {args.model_name}.vocab')
