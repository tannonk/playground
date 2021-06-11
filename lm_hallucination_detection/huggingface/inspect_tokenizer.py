#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from tokenizers.implementations import ByteLevelBPETokenizer
# from tokenizers.processors import BertProcessing
from tokenizers import Tokenizer
from datasets import load_dataset

dataset = load_dataset('text', data_files={
    'train': '/srv/scratch6/kew/lm_data/rrgen_de/raw/train.rev_resp',
    'test': '/srv/scratch6/kew/lm_data/rrgen_de/raw/test.rev_resp',
    'valid': '/srv/scratch6/kew/lm_data/rrgen_de/raw/valid.rev_resp'
    })

breakpoint()

tokenizer = Tokenizer.from_file("/srv/scratch6/kew/lm_data/rrgen_de/huggingface/tokenizer-de_rrgen.json")


# tokenizer.enable_truncation(max_length=512)

examples = [
    "Klar, teuer aber geht sogar für Züricher Verhältnisse, Essen ist super und Personal freundlich <BOS> Guten Tag N W Vielen herzlichen Dank, dass Sie bei uns im Zürcher Niederdörfli zu Gast waren und sich die Zeit für diese positive Bewertung genommen haben. Wir freuen uns sehr zu hören, dass es Ihnen bei uns gut gefallen hat und Sie vor allem unsere Schweizer Gerichte so richtig geniessen konnten. Wir bieten übrigens unter der Woche täglich wechselndes Mittagsmenüs zum absoluten Geniesserpreis an (weniger als 20 Franken inklusive Suppe oder Salat). Wir freuen uns auf jeden Fall alle schon jetzt, Sie ganz bald wieder bei uns zu begrüssen. Herzliche Grüsse, Ihr Swiss Chuchi Team Hotel Adler Zürich <EOS>",
    "Gut <endtitle> (+) Die gute Lage Sehr gute Frühstück Sehr freundliches Personal (-) Nicht angekündigte lärmige Bauarbeiten und das mit Baugerüst verhüllte Hotel. Balkon konnten wir nicht benutzen. <BOS> Guten Tag Estera Vielen Dank, dass Sie sich für unser familiengeführtes Hotel entschieden haben und Ihre Erfahrung mit uns teilen. Wir freuen uns auch zu hören, dass es Ihnen bei uns gut gefallen hat und Sie von unserer zentralen Lage profitieren konnten. Wie Sie auch auf unserer Homepage sehen, verschönern wir in der Tat noch bis Mitte Monat unser Hotel. Es tut uns sehr leid für etwaige Unannehmlichkeiten respektive für Störungen durch den Baulärm. Wir würden uns dennoch sehr darüber freuen, Sie bei Ihrem nächsten Aufenthalt wieder in unserem (verschönerten) Hotel begrüssen zu dürfen. Herzliche Grüsse, Ihr Hotel Drei Könige <EOS>"
    ]

