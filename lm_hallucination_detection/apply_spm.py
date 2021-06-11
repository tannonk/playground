#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for applying a trained sentencepiece model to a tokenized source text with corresponding token-level labels.

If using as regular sentencepiece apply script, i.e. raw data -> sp tokenized data, ignore `src_labels` argument.

python apply_spm.py --inpath /srv/scratch6/kew/lm_data/rrgen_de/raw/ --model /srv/scratch6/kew/lm_data/rrgen_de/sentencepiece.model --src rev_resp

"""

import argparse
from pathlib import Path
from typing import List, Tuple
import sentencepiece as sp
from tqdm import tqdm

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inpath', type=str, required=True, help='directory containing .review and .sentiment_seq files to apply sp model')
    ap.add_argument('--model', required=True, type=str, help='trained sentencepiece model')
    ap.add_argument('--src', required=True, type=str, help='src input file suffix')
    ap.add_argument('--tgt', required=False, type=str, help='tgt input file suffix')
    ap.add_argument('--src_labels', required=False, type=str, help='parallel annotation file for src sequence, e.g. token-level sentiment labels')
    
    # ap.add_argument('--test_pref', required=False, default='test', type=str, help='file prefix for test set')
    # ap.add_argument('--valid_pref', required=False, default='valid', type=str, help='file prefix for valid set')
    # ap.add_argument('--train_pref', required=False, default='train', type=str, help='file prefix for train set')
    # ap.add_argument('--symbols', required=False, type=str, help='path to txt file containing special symbols (expects one symbol per line).')
    
    return ap.parse_args()

def flatten(lst):
    return [item for sublist in lst for item in sublist]
    
def apply_sentencepiece_on_labeled_token_sequence(tok_seq: List[str], label_seq: List[str], spm) -> Tuple[str]:    
    
    assert len(tok_seq) == len(label_seq), f"[!] Input token sequence has different length to sentiment sequence\n{tok_seq}\n{label_seq}"

    sp_tok_seq = [spm.encode_as_pieces(tok) for tok in tok_seq]
    sp_label_seq = []
    
    for label, sp_tok in zip(label_seq, sp_tok_seq):
        sp_label_seq.append([label] * len(sp_tok))

    sp_tok_seq = flatten(sp_tok_seq)
    sp_label_seq = flatten(sp_label_seq)    

    assert len(sp_tok_seq) == len(sp_label_seq), f"[!] Output token sequence has different length to sentiment sequence\n{sp_tok_seq}\n{sp_label_seq}"
    
    return (' '.join(sp_tok_seq), ' '.join(sp_label_seq))

def apply_sentencepiece(line: str, spm):
    return ' '.join(spm.encode_as_pieces(line))

def process_parallel_files(review_infile, senti_infile, spm):
    """read in parallel review/sentiment files and process
    line by line.
    """
    review_outfile = str(review_infile)+'.sp'
    senti_outfile = str(senti_infile)+'.sp'

    # breakpoint()
    print(f'processing parallel {review_infile} -- {senti_infile}...')
    with open(review_outfile, 'w', encoding='utf8') as outf_rf, open(senti_outfile, 'w', encoding='utf8') as outf_sf:
        with open(review_infile, 'r', encoding='utf8') as inf_rf, open(senti_infile, 'r', encoding='utf8') as inf_sf:
            for review, sentiment in tqdm(zip(inf_rf, inf_sf)):
                review = review.strip().split()
                sentiment = sentiment.strip().split()                     
                review, sentiment = apply_sentencepiece_on_labeled_token_sequence(review, sentiment, spm)
                outf_rf.write(f'{review}\n')
                outf_sf.write(f'{sentiment}\n')
    return

def process_non_parallel_file(infile, spm):

    outfile = str(infile)+'.sp'
    
    print(f'processing {infile}...')
    
    with open(outfile, 'w', encoding='utf8') as outf:
        with open(infile, 'r', encoding='utf8') as inf:
            for line in tqdm(inf):
                line = line.strip()
                line = apply_sentencepiece(line, spm)
                outf.write(f'{line}\n')
    return

def test():
    bpemb_model = '/srv/scratch2/kew/sentencepiece_models/rrgen.de.bpe.vs10000.model'
    # bpemb_model = '/srv/scratch2/kew/embeddings/data/de/de.wiki.bpe.vs10000.model'
    spm = sp.SentencePieceProcessor(model_file=bpemb_model)
    # test
    t = 'hier nagt der zahn der zeit <endtitle> dem hotel würde eine renovierung gut tun , der teppichboden in den fluren im <digit> stock ist fleckig und ziemlich versifft . der teppichboden im zimmer war ein ähnliches hygienedebakel . besonders dreist war das ansinnen der rezeption , dass man das zimmer vor bezug und besichtigung bezahlen solle .'.split()
    s = 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O ROOM_NEUTRAL ROOM_NEUTRAL O O O O ROOM_POSITIVE ROOM_POSITIVE O O O O O O O O O O O O O O O O O O O O'.split()
    
    sp_t, sp_s = apply_sentencepiece_on_labeled_token_sequence(t, s, spm)

    # print(t, '\n', s)
    for x, y in zip(t, s):
        print(x,'-', y, end = ' ')

    for x, y in zip(sp_t.split(), sp_s.split()):
        print(x,'-', y, end = ' ')

    # print(sp_t, '\n', sp_s)

if __name__ == "__main__":
    # test()
    args = set_args()

    spm = sp.SentencePieceProcessor(model_file=args.model)
    
    inpath = Path(args.inpath)
    
    if 'RE_TEST' in str(inpath):
        print('Processing re:spondelligent subset for testing...')
        if args.src_labels and args.src:
            process_parallel_files(inpath / f're_test.{args.src}', inpath / f're_test.{args.src_labels}', spm)    
        elif args.src:
            process_non_parallel_file(inpath / f're_test.{args.src}', spm)
        if args.tgt:
            process_non_parallel_file(inpath / f're_test.{args.tgt}', spm)
            
    else:
        if args.src_labels and args.src:
            process_parallel_files(inpath / f'test.{args.src}', inpath / f'test.{args.src_labels}', spm)    
            process_parallel_files(inpath / f'valid.{args.src}', inpath / f'valid.{args.src_labels}', spm)
            process_parallel_files(inpath / f'train.{args.src}', inpath / f'train.{args.src_labels}', spm)
        elif args.src:
            process_non_parallel_file(inpath / f'test.{args.src}', spm)
            process_non_parallel_file(inpath / f'valid.{args.src}', spm)
            process_non_parallel_file(inpath / f'train.{args.src}', spm)
        
        if args.tgt:
            process_non_parallel_file(inpath / f'test.{args.tgt}', spm)
            process_non_parallel_file(inpath / f'valid.{args.tgt}', spm)
            process_non_parallel_file(inpath / f'train.{args.tgt}', spm)


    




    

