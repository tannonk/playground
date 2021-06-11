#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import json
import pandas as pd

SEED = 42

pickled_df = sys.argv[1]
output_dir = sys.argv[2]
prefix_reviews = True

Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(pickled_df)
print(f'Dataframe has {len(df)} entries...')

# ensure domain and rating match previous exp setups
df['domain'] = df['domain'].apply(lambda x: '<'+x.lower()+'>')
df['rating'] = df['rating'].apply(lambda x: '<'+str(x)+'>')

# TODO: replace rating values <= 0 with <1>!

if prefix_reviews:
    df['review'] = df.apply(lambda x: x.domain + ' ' + x.rating + ' ' + x.review, axis=1)
    print('added prefix tokens to review textm e.g.')
    print(df.iloc[10]['review'])

if 'split_imrg_compat' in df.columns:
    for split in ['train', 'test', 'valid']:
        split_df = df[df['split_imrg_compat'] == split]
        
        print(f'{split} has {len(split_df)} entries...')

        src_file = Path(output_dir) / f'{split}.review'
        tgt_file = Path(output_dir) / f'{split}.response'
        with open(src_file, 'w', encoding='utf8') as srcf:
            with open(tgt_file, 'w', encoding='utf8') as tgtf:
                # breakpoint()
                for row in split_df.itertuples(index=False):
                    srcf.write(f'{row.review}\n')
                    tgtf.write(f'{row.response}\n')

        # write out meta data to files as well
        split_df['domain'].to_csv(str(Path(output_dir) / f'{split}.domain'), index=False, header=False)
        split_df['rating'].to_csv(str(Path(output_dir) / f'{split}.rating'), index=False, header=False)
        split_df['source'].to_csv(str(Path(output_dir) / f'{split}.source'), index=False, header=False)

else: # dealing with new dataset from 2021_01 (use as additional eval set)
    # remove any empty reviews/responses
    df = df[df['review'] != '']
    df = df[df['response'] != '']

    # add source column
    df['source'] = 're'

    df = df.sample(n=450, random_state=SEED)

    file_stem = 'respo2021_01'

    print(f'Dataframe has {len(df)} entries...')
    src_file = Path(output_dir) / f'{file_stem}.review'
    tgt_file = Path(output_dir) / f'{file_stem}.response'
    with open(src_file, 'w', encoding='utf8') as srcf:
        with open(tgt_file, 'w', encoding='utf8') as tgtf:
            for row in df.itertuples(index=False):
                srcf.write(f'{row.review}\n')
                tgtf.write(f'{row.response}\n')
    
    # write out meta data to files as well
    df['domain'].to_csv(str(Path(output_dir) / f'{file_stem}.domain'), index=False, header=False)
    df['rating'].to_csv(str(Path(output_dir) / f'{file_stem}.rating'), index=False, header=False)
    df['source'].to_csv(str(Path(output_dir) / f'{file_stem}.source'), index=False, header=False)


print('done!')