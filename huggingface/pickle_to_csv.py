#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import json
import pandas as pd

pickled_df = sys.argv[1]
output_dir = sys.argv[2]

Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(pickled_df)
print(f'Dataframe has {len(df)} entries...')

if 'split_imrg_compat' in df.columns:
    for split in ['train', 'test', 'valid']:
        split_df = df[df['split_imrg_compat'] == split]
        split_df = split_df[['review', 'response']]
        output_file = Path(output_dir) / f'{split}.csv'
        split_df.to_csv(output_file, header=True, index=False)

else:
    # breakpoint()
    df = df[['review', 'response']]
    df = df[df['review'] != '']
    df = df[df['response'] != '']
    print(f'Dataframe has {len(df)} entries...')
    output_file = Path(output_dir) / 'respo2021_01.csv'
    df.to_csv(output_file, header=True, index=False)

print('done!')