#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_data, clean_unnecessary_spaces

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

# Google Data
train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/dev.tsv", sep="\t").astype(str)

train_df = train_df.loc[train_df["label"] == "1"]
eval_df = eval_df.loc[eval_df["label"] == "1"]

train_df = train_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)
eval_df = eval_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)

train_df = train_df[["input_text", "target_text"]]
eval_df = eval_df[["input_text", "target_text"]]

train_df["prefix"] = "paraphrase"
eval_df["prefix"] = "paraphrase"

# # MSRP Data
# train_df = pd.concat(
#     [
#         train_df,
#         load_data("data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"),
#     ]
# )
# eval_df = pd.concat(
#     [
#         eval_df,
#         load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),
#     ]
# )

# Quora Data

# The Quora Dataset is not separated into train/test, so we do it manually the first time.
df = load_data(
    "data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate"
)
q_train, q_test = train_test_split(df)

q_train.to_csv("data/quora_train.tsv", sep="\t")
q_test.to_csv("data/quora_test.tsv", sep="\t")

# The code block above only needs to be run once.
# After that, the two lines below are sufficient to load the Quora dataset.

# q_train = pd.read_csv("data/quora_train.tsv", sep="\t")
# q_test = pd.read_csv("data/quora_test.tsv", sep="\t")

train_df = pd.concat([train_df, q_train])
eval_df = pd.concat([eval_df, q_test])

train_df = train_df[["prefix", "input_text", "target_text"]]
eval_df = eval_df[["prefix", "input_text", "target_text"]]

train_df = train_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)

eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

print(train_df)

train_df.to_pickle('./data/prep/train.pkl')
eval_df.to_pickle('./data/prep/eval.pkl')