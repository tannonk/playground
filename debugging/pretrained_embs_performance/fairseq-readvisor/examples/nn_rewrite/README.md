# Don't write - *Re*write!

This module contains a seq2seq rewrite model based on the
Transformer (Vaswani et al. 2017) following the approach 
proposed by Hossain et al. 2020.

Review responses typically have a formulaic structure, e.g.
    
    1. Greeting
    2. Thanking the reviewer
    3. The body
    4. Closing
    5. Salutation

The body is essence of the response and typically addresses
points mentioned in the review. This part of the response
should be suitable and individualised for the given review 
and not too general. Relevant details about the
establishment  could also be included such as the history, 
design, menu items, specials, etc.

Given this general structure, we can potentially leverage handwritten 
review responses from the training data and think of them as 
"soft templates" (Cao et al. 2018), which the model can 
learn to rewrite or edit into a target response (Y).


## The Model

The model extends Fairseq's seq2seq transformer model
(Vaswani et al. 2017) based on the approach presented by
Hossain et al. 2020.

The main change is the addition of BERT-like segment
embeddings that allow the model to distinguish between 
two unique inputs, e.g. one input review (x) and a 
semantically similar handwritten response from the 
training data (y').

To avoid altering the default translation task and the
language-pair dataset, we simply generate the segment
embeddings on the fly, similar to how the positional token
embeddings are generated. 

We use the seperator token specified as a command-line
argument to calculate a masking vector for the concatenated
input sequence (x ; y') and then convert all zero values to
their relevant segment ID.

---

## Relevant Papers

- Cao et al. (2018) Retrieve, Rerank and Rewrite: Soft
Template Based Neural Summarization

- Hossain et al. (2020) Simple and Effective
  Retrieve-Edit-Rerank Text Generation

- Vaswani et al. (2017) Attention is All you Need

---


Author: Tannon Kew
Date: 16.12.2020