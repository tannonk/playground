#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
converts jsonl format to simple src <BOS> hyp format

Example call

    python prep_data_for_scoring.py /srv/scratch6/kew/mbart/hospo_respo/respo_final/2021_06/mbart_model_2021-06-18/ft/2021-06-20_14-52-42/inference/translations_21.json /srv/scratch6/kew/mbart/hospo_respo/respo_final/2021_06/data/test.lang_tags de_DE > /srv/scratch6/kew/mbart/hospo_respo/respo_final/2021_06/mbart_model_2021-06-18/ft/2021-06-20_14-52-42/inference/translations_21.simple.txt

"""
import json
import sys

infile = sys.argv[1]
try:
    lang_tags = sys.argv[2]
    lang = sys.argv[3]
except:
    lang_tags = None

if lang_tags and lang:
    with open(lang_tags, 'r', encoding='utf8') as f:
        lang_tags = [line.strip() for line in f]

# print(lang_tags)
# breakpoint()

with open(infile, 'r', encoding='utf8') as f:
    for i, line in enumerate(f):
        line = json.loads(line.strip())
        if lang_tags and lang:
            if lang_tags[i] == lang:
                print(f"{line['src']} <BOS> {line['hyps'][0]['hyp']}")
            else:
                pass
        else:
            print(f"{line['src']} <BOS> {line['hyps'][0]['hyp']}")
