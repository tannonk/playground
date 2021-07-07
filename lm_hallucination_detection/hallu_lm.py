#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

# minimal example
    python hallu_lm.py --dict /srv/scratch6/kew/lm_data/rrgen_de/gpt2tiny/lm_cond/ --lm /srv/scratch6/kew/lm_data/rrgen_de/gpt2tiny/lm_cond/210615/checkpoint_best.pt > data_egs/scores.jsonl

python hallu_lm.py --dict /srv/scratch6/kew/lm_data/rrgen_de/gpt2tiny/lm_cond/ --lm /srv/scratch6/kew/lm_data/rrgen_de/gpt2tiny/lm_cond/210615/checkpoint_best.pt --infile /srv/scratch6/kew/lm_data/rrgen_de/validation_src_tgt_for_hallucination_detection_de.txt --outfile /srv/scratch6/kew/lm_data/rrgen_de/validation_src_tgt_hallucination_detection_de.jsonl

"""

import sys
import math
from typing import List, Tuple, Dict
import logging
import numpy as np
from tqdm import tqdm
import json
import re
import pandas as pd
import torch
import argparse
# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel


def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--dict', type=str, help='path to data directory containing `dict.txt` file required by fairseq.')
    ap.add_argument('--lm', type=str, required=True, help='path to conditional language model checkpoint')
    ap.add_argument('--gpu', action='store_true', help='if specified, model will be loaded onto first GPU')
    ap.add_argument('--infile', type=str, required=False, help='optional infile for processing/scoring source-target pairs')
    ap.add_argument('--outfile', type=str, required=False, help='optional outfile for writing json outputs.')
    
    return ap.parse_args()

def iter_lines(infile):
    with open(infile, 'r', encoding='utf8') as inf:
        for line in inf:
            yield line.strip()
            
def merge_subword_scores(sw_scores, sw_token_string):
    # breakpoint()
    new_string = ''
    new_scores = []
    for i, tok in enumerate(sw_token_string.split()):
#         print(new_scores)
        if tok[0] != '▁' and i != 0:            
#             print(i, tok)
            new_scores[-1] += sw_scores[i]
            new_string += tok
        else:
            new_scores.append(sw_scores[i])
            new_string += tok.replace('▁', ' ')
            
    return np.array(new_scores), new_string

class HalluLMScorer:

    """
    Implementation of LM-based hallucination detection
    inspired by Filippova (2020) / Fernandes et al. (2021).
    """

    def __init__(self, dictionary: str, LM_path: str, use_gpu: bool):
        
        self.lm = self._load_custom_lm(dictionary, LM_path) 
        # disable dropout
        self.lm.eval()

        if torch.cuda.is_available() and use_gpu:
            logging.info(torch.cuda.current_device())
            self.lm.cuda() # move model to GPU
    
    def score_lm(self, target: str, context: str, verbose=False) -> torch.Tensor:
        """
        Use language model to score the target sequence with
        access to source context
        
        Args:
            target: target text or model hypothesis
            context: expected to be consist only of start of sequence token `<BOS>`
        """
        # encode adds </s> implicitly, so remove this from context
        context_tokens = self.lm.encode(context)[:-1] 
        context_length = context_tokens.shape[-1]
        target_tokens = self.lm.encode(target)

        if verbose:
            logging.info(
                f'prefix = {self.lm.tgt_dict.string(context_tokens)} ~~~ target = {self.lm.tgt_dict.string(target_tokens)}')
        
        # breakpoint()
        if len(context_tokens) + len(target_tokens) > self.lm.max_positions:
            logging.warning('could not score item due to length... (To do: implement truncation or use a smaller model.)')
            return (None, None)
            # if len(context_tokens) >= self.lm.max_positions:
                # truncate context
            # breakpoint()

        norm_scores = self.lm.score(context+target)
        # breakpoint()
        # trim scores corresponding to <BOS> token
        return (
            norm_scores['positional_scores'][context_length:].cpu().numpy(), 
            self.lm.string(norm_scores['tokens'][context_length:])
        )

    def score_sequence(self, sequence):
        
        context, target = re.split(r'\s?<BOS>\s?', sequence)
            
        lm_scores, lm_tokens = self.score_lm(target, '<BOS>')
        lmx_scores, lmx_tokens = self.score_lm(target, context+' <BOS>')

        lm_scores, lm_tokens = merge_subword_scores(lm_scores, lm_tokens)
        lmx_scores, lmx_tokens = merge_subword_scores(lmx_scores, lmx_tokens)

        assert lm_tokens == lmx_tokens
        assert len(lm_scores) == len(lmx_scores)


        # formula (2) from Filippova (2020) (**modified**)
        # NOTE: for each token Wyt, check if pLM(Wyt) > pLMx(Wyt)
        # number of tokens for which lm score is greater than lmx
        I = lm_scores > lmx_scores
        sum_score = I.sum()/len(lm_scores)
        # breakpoint()

        return {
            "source_text": context,
            "target_text": target,
            "hal-lm_score": sum_score,
            "diff_pos_scores": (lm_scores - lmx_scores).tolist(),
            "bool": I.tolist(),
            "target_tokens": lmx_tokens,
        }

    def _load_custom_lm(self, dictionary: str, checkpoint: str):
        return TransformerLanguageModel.from_pretrained(dictionary, checkpoint, bpe='sentencepiece')

    def truncate_input(self, input):
        pass



if __name__ == '__main__':

    args = set_args()
    hlm = HalluLMScorer(args.dict, args.lm, args.gpu)

    if args.infile:
        cond_responses = iter_lines(args.infile)
    else:
        cond_responses = [
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unserem Heimat Burger.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unseren Speisen.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unseren Thai-Gerichten.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unseren authentischen Thai-Gerichten.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unserer Holzofen-Pizza.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unserer traditionellen Holzofen-Pizza.",
            
        ]

    if not args.outfile:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile, 'w', encoding='utf8')
    
    # scores = []
    for text in tqdm(cond_responses):
        try:
            score_dict = hlm.score_sequence(text)
            json.dump(score_dict, outfile, ensure_ascii=False)
            outfile.write('\n')
            # scores.append(score_dict)
        except:
            continue

    # breakpoint()
    # df = pd.DataFrame(scores)
    # print(df.to_csv())

    outfile.close()

