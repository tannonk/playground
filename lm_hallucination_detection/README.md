
This repo contains LM-based methods for hallucination detection in
conditioned text generation.

## LMcond vs LMunc

This idea comes from Filippova (2020) "Controlled
Hallucinations: Learning to Generate Faithfully from Noisy
Data".

The basic ideas is as follows:

Train two language models LM_{cond} and LM_{unc} to predict the target
texts. 

LM_{unc} is an unconditional LM trained only on the targets.


LM_{cond} is a conditional LM trained on source-target pairs.

is contains scripts for training and evaluating a transformer language model with Fairseq.

## Minimal example

`LMcond` format: SRC + TGT
```
Had a great time at Hotel Adler <s> Dear John, thank you for your review. We're glad you enjoyed your visit to the Hotel Adler in Zurich. </s>
```

`lmunc` format: TGT
```
<s> Dear John, thank you for your review. We're glad you enjoyed your visit to the Hotel Adler in Zurich. </s>
```

## Setup

To set up the virtual environment, run

```
cd scripts
bash setup_env
conda activate <name-of-new-env>
```

Fairseq commands are taken from
https://github.com/pytorch/fairseq/tree/master/examples/language_model

## Build LMs with Fairseq

### Data prep
- ensure that `<s>` and `</s>` are inserted correctly in
  training data files
- use `train_spm.py` to learn a SentencePiece model on the data
- use `apply_spm.py` to tokenize data with the SentencePiece
  model
- `prepare_lmunc_lmcond_data.sh` calls `fairseq-preprocess`
  for both datasets

### Train
- `train_lmunc` can be used to train the unconditional LM.
- `train_lmcond.sh` can be used to train the conditional LM

### Evaluate
- 
- 

