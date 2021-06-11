#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

python load_lm.py /srv/scratch6/kew/lm_data/rrgen_de/rev_resp.sp-lm-data-bin --chkpt /srv/scratch6/kew/lm_data/rrgen_de/lm_cond/checkpoint_best.pt

"""

import torch
import argparse
# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel

unc_responses = [
        "<BOS> Vielen Dank für Ihre Bewertung. <EOS>",
        "<BOS> Wir freuen uns sehr, dass Sie einen sehr schönen Aufenthalt bei uns hatten und freuen uns schon jetzt auf das nächste Wiedersehen! Viele Grüße, Ihr Team vom Berliner Fernsehturm <EOS>",
        "<BOS> Sehr geerter A8627VEsusang, es freut mich zu lesen, dass wir wohnqualitativ für Ihre Familie optimal sind und hoffen Sie auch weiterhin in Adina´s in Hamburg, Frankfurt, Berlin, Kopenhagen und Budapest begrüßen zu dürfen. Bis zum nächsten Mal. Mit freundlichen Grüßen Annette Jost <EOS>",
        "<BOS> Vielen Dank für Ihre Bewertung und Weiterempfehlung! Wir freuen uns, dass es Ihnen sehr gut gefallen hat und hoffen, dass Sie in Zukunft auch einmal unser zweites Restaurant - das Gourmet 1895 - besuchen. Mit Vorfreude auf ein Wiedersehen senden wir herzliche Grüße aus dem Hotel Kaiserhof Münster! <EOS>",
        "<BOS> Liebe Tanja M Herzlichen Dank für das tolle Feedback und die Vergabe von 5 Sternen für unseren HEIMAT Burger und die Pommes. Wir freuen uns, wenn wir Dich schon bald wieder bei uns in der HEIMAT Küche + Bar begrüßen dürfen. Herzliche Grüße aus der Hafencity Dominique Alexander Ewerth F&B Manager 25hours Hotel HafenCity 25hours Hotel Altes Hafenamt <EOS>",
    ]

cond_responses = [
    "Sehr schön! <BOS> Vielen Dank! <EOS>",
    "Überraschung im Dortmund <endtitle> Wir hatten von Dortmund nicht allzu viel erwartet und waren umso überraschter dort ein so schönes Hotel vorzufinden. Sehr guter Service am Empfang, tolles, modernes Zimmer. Das Bad war nicht riesig aber gut durchdacht und äußerst sauber und modern. Das Restaurant ist gemütlich modern eingerichtet. Beim Frühstück gab es reichlich Auswahl, so dass Frühstücken wirklich Spaß gemacht hat. <BOS> Vielen Dank für Ihre Bewertung. <EOS>",
    "Super schönes Erlebnis, ich werde es sehr positiv in Erinnerung behalten. <endtitle> Der Besuch war sehr schön und total romantisch. Wir haben knapp 95 € bezahlt, mit Eintritt und Tischreservierung lagen wir bei 145 Euro. Die Preise waren verständlicherweise etwas höher, das Essen war schön aufbereitet und auch sehr lecker, für den Preis jedoch nicht überdurchschnittlich gut. Bei der Vorspeise hätte ich mir gerne frisches Gemüse gewünscht, statt eingelegtes und das Fleisch war ein Tick zu trocken. Wer vorher shoppen war, sollte bedenken, dass man leider keine Taschen abgeben kann. Meckern auf hohem Niveau. Alles in allem gut. :) <BOS> Wir freuen uns sehr, dass Sie einen sehr schönen Aufenthalt bei uns hatten und freuen uns schon jetzt auf das nächste Wiedersehen! Viele Grüße, Ihr Team vom Berliner Fernsehturm <EOS>",
    "tolles Hotel <endtitle> Das Konzept ist super, wir sind nun schon mit drei Kindern als auch allein zu Besuch in dieser Hotelkette gewesen, vor allem für die erste Variante ist es schwer, etwas wohnqualitativ vergleichbares zu finden. <BOS> Sehr geerter A8627VEsusang, es freut mich zu lesen, dass wir wohnqualitativ für Ihre Familie optimal sind und hoffen Sie auch weiterhin in Adina´s in Hamburg, Frankfurt, Berlin, Kopenhagen und Budapest begrüßen zu dürfen. Bis zum nächsten Mal. Mit freundlichen Grüßen Annette Jost <EOS>",
    "schönes Ambiente <endtitle> Zum Hotel Kaiserhof zugehöriges Restaurant. Vielfältige Auswahl, sehr lecker. Preisniveau im oberen Segment. Service hellwach, nett und zuvorkommend. Einen Besuch wert. <BOS> Vielen Dank für Ihre Bewertung und Weiterempfehlung! Wir freuen uns, dass es Ihnen sehr gut gefallen hat und hoffen, dass Sie in Zukunft auch einmal unser zweites Restaurant - das Gourmet 1895 - besuchen. Mit Vorfreude auf ein Wiedersehen senden wir herzliche Grüße aus dem Hotel Kaiserhof Münster! <EOS>",
    "Sehr lecker und üppig <endtitle> Der Heimat Burger ist sehr lecker, die Pommes dazu außergewöhnlich gut. Das Essen darf als Hotelgast mit aufs Zimmer genommen werden <BOS> Liebe Tanja M Herzlichen Dank für das tolle Feedback und die Vergabe von 5 Sternen für unseren HEIMAT Burger und die Pommes. Wir freuen uns, wenn wir Dich schon bald wieder bei uns in der HEIMAT Küche + Bar begrüßen dürfen. Herzliche Grüße aus der Hafencity Dominique Alexander Ewerth F&B Manager 25hours Hotel HafenCity 25hours Hotel Altes Hafenamt <EOS>",    
]



def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('data_dir', type=str, help='path to data dir containing `dict.txt` file required by fairseq.')
    ap.add_argument('--chkpt', type=str, required=True, help='path to fairseq model checkpoint')
    # ap.add_argument('--spm', type=str, required=True, help='path to spm model')
    return ap.parse_args()

def score(lm, sentence, prefix=None):

    breakpoint()

    prefix_tokens = lm.encode(prefix)
    print('Prefix encoded as:', lm.tgt_dict.string(prefix_tokens))
    prefix_length = prefix_tokens.shape[-1]
    norm_scores = lm.score(prefix+sentence)
    
    # select only scores for sentence tokens
    norm_scores['positional_scores'][prefix_length:]

    # else:
    #     norm_scores = lm.score(sentence)
    print(norm_scores)
    # prefix_tokens = lm.encode(pref)
    # prefix_size = prefix_tokens.shape[-1]
    # sentence_tokens = lm.encode(sentence)


    # inference_step_args = {'prefix_tokens': prefix_tokens}
    # lm.generate(
    #     sentence_tokens,
    #     beam=1,
    #     verbose=True,
    #     skip_invalid_size_inputs=True,
    #     prefix_size=prefix_size
    #     )

    # from hugging face:
    # for i in range(1, len(tokenize_input)+1):
    #     tensor_input = torch.tensor([tokenize_input[:i]])
    #     print(tensor_input)
    #     loss=model(tensor_input, labels=tensor_input)[0]
    #     print(np.exp(loss.detach().numpy()))
    # return np.exp(loss.detach().numpy())

if __name__ == '__main__':

    # lm =
    # TransformerLanguageModel.from_pretrained('/srv/scratch6/kew/lm_data/wikitext-2/data-bin',
    # '/srv/scratch6/kew/lm_data/wikitext-2/checkpoints/transformer_wikitext-2/checkpoint_best.pt',
    # tokenizer='moses')
    args = set_args()
    lm = TransformerLanguageModel.from_pretrained(args.data_dir, args.chkpt, bpe='sentencepiece')

    # breakpoint()
    lm.eval()  # disable dropout
    if torch.cuda.is_available():
        # print(torch.cuda.current_device())
        lm.cuda() # move model to GPU
    
    # breakpoint()

    for text in cond_responses:
        pref, sentence = text.split('<BOS>')
        pref = pref + ' <BOS>'
        sentence = sentence
        print('PREF', pref)
        print('SENT', sentence)
        score(lm, sentence, pref)

    

    # lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp()
    # # tensor(15.1474)

    # lm.sample('Barack Obama', beam=5)
    # "Barack Obama (...)"








# # List available models
# torch.hub.list('pytorch/fairseq')  # [..., 'transformer_lm.wmt19.en', ...]

# # Load an English LM trained on WMT'19 News Crawl data
# en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')
# en_lm.eval()  # disable dropout

# # Move model to GPU
# en_lm.cuda()

# # Sample from the language model
# en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
# # "Barack Obama is coming to Sydney and New Zealand (...)"

# # Compute perplexity for a sequence
# en_lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp()
# # tensor(15.1474)
