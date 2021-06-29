#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This implements the idea from Filippova (2020), where we use
TWO language models (one conditional, one uncondintional) to
score a sequence to compute degree of hallucination (i.e.
the amount that the unconditional LM loss for the next token
is less than that of the LMx loss).

NOTE: Fernandes et al. (2021) / Bugliarello et al. (2020)
describe a similar idea with Condition Cross-Mutual
Information. Important here, however, is to use the SAME
model for both LM and LMx --> see `hallu_lm.py` for
implementation based on using the SAME LM for both.

python hallu_2lm.py --dict /srv/scratch6/kew/lm_data/rrgen_de/lm_cond/ --lm /srv/scratch6/kew/lm_data/rrgen_de/lm_unc/210615/checkpoint_best.pt --lmx /srv/scratch6/kew/lm_data/rrgen_de/lm_cond/210615/checkpoint_best.pt --infile /srv/scratch6/kew/lm_data/rrgen_de/validation_src_tgt_for_hallucination_detection_de.txt --outfile /srv/scratch6/kew/lm_data/rrgen_de/validation_src_tgt_hallucination_detection_de.jsonl

"""

import sys
from tqdm import tqdm
import json
import re
import torch
import argparse
# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel

def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--dict', type=str, help='path to data directory containing `dict.txt` file required by fairseq.')
    ap.add_argument('--lm', type=str, required=True, help='path to unconditional language model dir')
    ap.add_argument('--lmx', type=str, required=True, help='path to conditional language model dir')
    ap.add_argument('--infile', type=str, required=False, help='optional infile for processing/scoring source-target pairs')
    ap.add_argument('--outfile', type=str, required=False, help='optional outfile for writing json outputs.')
    
    return ap.parse_args()

class HalluLMScorer:

    """
    Implementation of LM-based hallucination detection
    proposed by Filippova (2020).

    """

    def __init__(self, dictionary: str, LM_path: str, LMx_path: str,):
        
        self.lm = self._load_custom_lm(dictionary, LM_path)
        self.lmx = self._load_custom_lm(dictionary, LMx_path)
        
        # disable dropout
        self.lm.eval()
        self.lmx.eval()

        # if torch.cuda.is_available():
        # # print(torch.cuda.current_device())
        #     self.lm.cuda() # move model to GPU
        #     self.lmx.cuda()

    def score_lmx(self, target: str, context: str, verbose=False) -> torch.Tensor:
        """
        Use the context-aware conditional language model
        (LMx) to score the target sequence.
        
        Args:
            target: target text or model hypothesis
            context: expected to be consist of original
            source text + start of sequence token `<BOS>`
        """
        # encode adds </s> implicitly, so remove this from context
        context_tokens = self.lmx.encode(context)[:-1] 
        context_length = context_tokens.shape[-1]
                 
        if verbose:
            print(
                f'prefix = {self.lmx.tgt_dict.string(context_tokens)} ~~~ target = {self.lmx.tgt_dict.string(self.lmx.encode(target))}')
        
          
        norm_scores = self.lmx.score(context+target)
        # trim scores corresponding to context and <BOS> seperator token
        return (
            norm_scores['positional_scores'][context_length:].numpy(), 
            self.lmx.string(norm_scores['tokens'][context_length:])
        )
    
    def score_lm(self, target: str, context: str, verbose=False) -> torch.Tensor:
        """
        Use the unconditional language model (LM) to score
        the target sequence.
        
        Args:
            target: target text or model hypothesis
            context: expected to be consist only of start of sequence token `<BOS>`
        """
        # encode adds </s> implicitly, so remove this from context
        context_tokens = self.lm.encode(context)[:-1] 
        context_length = context_tokens.shape[-1]

        if verbose:
            print(
                f'prefix = {self.lm.tgt_dict.string(context_tokens)} ~~~ target = {self.lm.tgt_dict.string(self.lm.encode(target))}')
        
        norm_scores = self.lm.score(context+target)
        # trim scores corresponding to <BOS> token
        return (
            norm_scores['positional_scores'][context_length:].numpy(), 
            self.lm.string(norm_scores['tokens'][context_length:])
        )

    def score_sequence(self, sequence):
        context, target = re.split(r'\s?<BOS>\s?', sequence)
            
        
        lm_scores, lm_tokens = self.score_lm(target, '<BOS>')
        lmx_scores, lmx_tokens = self.score_lmx(target, context+' <BOS>')

        assert lm_tokens == lmx_tokens
        assert len(lm_scores) == len(lmx_scores)

        # NOTE: for each token Wyt, check if pLM(Wyt) > pLMx(Wyt)
        # number of tokens for which lm score is greater than lmx
        I = lm_scores > lmx_scores
        # breakpoint()

        sum_score = I.sum()/len(lm_scores)

        # return (1-sum_score, lm_scores - lmx_scores, tokens)
        return {
            "source_text": context,
            "target_text": target,
            "hal-lm_score": sum_score,
            "diff_pos_scores": (lm_scores - lmx_scores).tolist(),
            "target_tokens": lmx_tokens,
        }

    def _load_custom_lm(self, dictionary: str, checkpoint: str):
        return TransformerLanguageModel.from_pretrained(dictionary, checkpoint, bpe='sentencepiece')        


def iter_lines(infile):
    with open(infile, 'r', encoding='utf8') as inf:
        for line in inf:
            yield line.strip()
            


if __name__ == '__main__':

    args = set_args()
    hlm = HalluLMScorer(args.dict, args.lm, args.lmx)

    if args.infile:
        cond_responses = iter_lines(args.infile)
    else:
        cond_responses = [
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unserem Heimat Burger.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unseren traditionellen Thai-Gerichten.",
            "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. <BOS> Liebe Tanja Herzlichen Dank für das tolle Feedback zu unserer Holzofenpizza.",
        ]

    if not args.outfile:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile, 'w', encoding='utf8')
    
    for text in tqdm(cond_responses):
        score_dict = hlm.score_sequence(text)
        json.dump(score_dict, outfile, ensure_ascii=False)
        outfile.write('\n')

    outfile.close()
