#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python hallu_lm.py --lm /srv/scratch6/kew/lm_data/rrgen_de/lm_unc/ --lmx /srv/scratch6/kew/lm_data/rrgen_de/lm_cond

TODO: fix sep token between context and target in tokenizer

"""

import os
from pathlib import Path
import torch
import argparse
# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel


class HalluLMScorer:

    """
    Implementation of LM-based hallucination detection
    proposed by Filippova (2020).

    """

    def __init__(self, LM_path: str, LMx_path: str,):

        self.lm = self._load_custom_lm(LM_path)
        self.lmx = self._load_custom_lm(LMx_path)
        
        # disable dropout
        self.lm.eval()
        self.lmx.eval()

        # if torch.cuda.is_available():
        # # print(torch.cuda.current_device())
        #     self.lm.cuda() # move model to GPU
        #     self.lmx.cuda()

    def score_lmx(self, context: str, target: str, verbose=False) -> torch.Tensor:
        """
        Use the context-aware conditional language model
        (LMx) to score the target sequence
        """
        context_tokens = self.lmx.encode(context)
        
        if verbose:
            print(
                f'prefix = {self.lmx.tgt_dict.string(context_tokens)} ~~~ target = {self.lmx.tgt_dict.string(self.lmx.encode(target))}')
        
        context_length = context_tokens.shape[-1]
        norm_scores = self.lmx.score(context+target)
    
        # trim scores for context tokens
        return norm_scores['positional_scores'][context_length:]
    
    def score_lm(self, target: str, verbose=False) -> torch.Tensor:
        """
        Use the unconditional language model (LM) to score the target sequence
        """
        if verbose:
            print(
                f'target = {self.lmx.tgt_dict.string(self.lmx.encode(target))}')
        
        norm_scores = self.lm.score(target)
        # trim score for <BOS> token
        return norm_scores['positional_scores'][1:]

    def score_sequence(self, sequence):
        context, target = sequence.split('<BOS>')
            
        target = '<BOS> ' + target
        # print('CONTEXT:', context, 'TARGET:', target)
        lm_scores = self.score_lm(target)
        lmx_scores = self.score_lmx(context, target)

        # print(len(lm_scores), len(lmx_scores))
        assert len(lm_scores) == len(lmx_scores)

        # NOTE: for each token Wyt, check if pLM(Wyt) > pLMx(Wyt)

        I = lm_scores > lmx_scores # number of tokens for which lm score is greater than lmx
        sum_score = I.sum().item()/len(lm_scores)
        # breakpoint()

        return (sum_score, lm_scores - lmx_scores)

    def _load_custom_lm(self, path: str):
        checkpoint = os.path.join(path, 'checkpoint_best.pt')
        return TransformerLanguageModel.from_pretrained(path, checkpoint, bpe='sentencepiece')        


def set_args():
    ap = argparse.ArgumentParser()
    
    # ap.add_argument('data_dir', type=str, help='path to data dir containing `dict.txt` file required by fairseq.')
    ap.add_argument('--lm', type=str, required=True, help='path to unconditional language model dir')
    ap.add_argument('--lmx', type=str, required=True, help='path to conditional language model dir')
    
    return ap.parse_args()


cond_responses = [
    "Sehr schön! <BOS> Vielen Dank! <EOS>",
    "Überraschung im Dortmund <endtitle> Wir hatten von Dortmund nicht allzu viel erwartet und waren umso überraschter dort ein so schönes Hotel vorzufinden. Sehr guter Service am Empfang, tolles, modernes Zimmer. Das Bad war nicht riesig aber gut durchdacht und äußerst sauber und modern. Das Restaurant ist gemütlich modern eingerichtet. Beim Frühstück gab es reichlich Auswahl, so dass Frühstücken wirklich Spaß gemacht hat. <BOS> Vielen Dank für Ihre Bewertung. <EOS>",
    "Super schönes Erlebnis, ich werde es sehr positiv in Erinnerung behalten. <endtitle> Der Besuch war sehr schön und total romantisch. Wir haben knapp 95 € bezahlt, mit Eintritt und Tischreservierung lagen wir bei 145 Euro. Die Preise waren verständlicherweise etwas höher, das Essen war schön aufbereitet und auch sehr lecker, für den Preis jedoch nicht überdurchschnittlich gut. Bei der Vorspeise hätte ich mir gerne frisches Gemüse gewünscht, statt eingelegtes und das Fleisch war ein Tick zu trocken. Wer vorher shoppen war, sollte bedenken, dass man leider keine Taschen abgeben kann. Meckern auf hohem Niveau. Alles in allem gut. :) <BOS> Wir freuen uns sehr, dass Sie einen sehr schönen Aufenthalt bei uns hatten und freuen uns schon jetzt auf das nächste Wiedersehen! Viele Grüße, Ihr Team vom Berliner Fernsehturm <EOS>",
    "tolles Hotel <endtitle> Das Konzept ist super, wir sind nun schon mit drei Kindern als auch allein zu Besuch in dieser Hotelkette gewesen, vor allem für die erste Variante ist es schwer, etwas wohnqualitativ vergleichbares zu finden. <BOS> Sehr geerter A8627VEsusang, es freut mich zu lesen, dass wir wohnqualitativ für Ihre Familie optimal sind und hoffen Sie auch weiterhin in Adina´s in Hamburg, Frankfurt, Berlin, Kopenhagen und Budapest begrüßen zu dürfen. Bis zum nächsten Mal. Mit freundlichen Grüßen Annette Jost <EOS>",
    "schönes Ambiente <endtitle> Zum Hotel Kaiserhof zugehöriges Restaurant. Vielfältige Auswahl, sehr lecker. Preisniveau im oberen Segment. Service hellwach, nett und zuvorkommend. Einen Besuch wert. <BOS> Vielen Dank für Ihre Bewertung und Weiterempfehlung! Wir freuen uns, dass es Ihnen sehr gut gefallen hat und hoffen, dass Sie in Zukunft auch einmal unser zweites Restaurant - das Gourmet 1895 - besuchen. Mit Vorfreude auf ein Wiedersehen senden wir herzliche Grüße aus dem Hotel Kaiserhof Münster! <EOS>",
    "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. Das Essen darf als Hotelgast mit aufs Zimmer genommen werden <BOS> Liebe Tanja M Herzlichen Dank für das tolle Feedback und die Vergabe von 5 Sternen für unseren HEIMAT Burger und die Pommes. Wir freuen uns, wenn wir Dich schon bald wieder bei uns in der HEIMAT Küche + Bar begrüßen dürfen. Herzliche Grüße aus der Hafencity Dominique Alexander Ewerth F&B Manager 25hours Hotel HafenCity 25hours Hotel Altes Hafenamt <EOS>",    
]

if __name__ == '__main__':

    args = set_args()
    hlm = HalluLMScorer(args.lm, args.lmx)
    for text in cond_responses:
        print(hlm.score_sequence(text))

    