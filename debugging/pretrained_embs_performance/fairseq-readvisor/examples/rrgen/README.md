# Review Response Generation for Online Hospitality Reviews

This module contains the source code (model, task, architecture and
supporting scripts) for review response generation for the hospitality domain. This
bulk of this work was done as part of the ReAdvisor project
at the University of Zurich 2020/2021. 

This implementation is based on the paper 'Automating App
Review Response Generation' by Gao et al. 2019
[https://ieeexplore.ieee.org/document/8952476]. The main
idea is to extend the basic encoder-decoder architecture,
popularised by MT, with additional attribute features
suitable for the task of seq2seq-based response generation.

There are some slight differences to the implementation proposed by
Gao et al.:

1. We use LSTMs instead of GRUs
2. We do not make use of the keyword component since it is
   shown to bring little gain in performance in the original
   authors' ablation study.
3. We do not implement the explicit review length component (yet)
4. Since our additional attributes are valid vectors, e.g. a
   category label {0, 1}, a review rating value {1, 5} and a
   review sentiment score (either {1, 10} of a custom
   aspect-level sentiment vector), we do not project these
   values through a learned embedding layer.
5. We use a custom sentiment engine which provides
   aspect-level sentiment analysis on input reviews in the
   form of a 25-d vector (a flattened 5x5 matrix).

### A Word on Decoding

Instead of modifying (and potentially breaking) Fairseq's
SequenceGenerator found in `sequence_generator.py`, we
replace it with a task-specific SequenceGenerator class
defined in `sequence_generator_rrgen.py`.

SequenceGenerator is called in the rrgen_translation task,
which contains a method `build_generator`, overriding the
default `build_generator` method, which is inherited from
the `fairseq_task` base class.

NB. funtionality is essentially the same in both
`build_generator` methods. Only the import statement changes
for the initialisation of the SequenceGenerator object.

---

## Changes

1. For experiments performed before January 2021, the
optimizer parameter was set by default with

```
@register_model_architecture('rrgen_lstm', 'rrgen_lstm_arch')
def rrgen_lstm_arch(args):
   ...
```

Now, optimizer needs to be passed explicitly with `fairseq-train`.

---

Author: Tannon Kew

Email: kew@cl.uzh.ch

Date: 02.01.2021