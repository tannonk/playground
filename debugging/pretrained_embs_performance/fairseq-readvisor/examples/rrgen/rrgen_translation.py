#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

Modification of Fairseq's original tranlsation task for
RRGen data format.

Date: 18 Aug 2020
Author: Tannon Kew
Email: kew@cl.uzh.ch

"""


from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, search, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import LegacyFairseqTask, register_task
# custom dataset
from .rrgen_dataset import RRGenDataset

# for custom dictionaries
from fairseq.file_io import PathManager
from collections import Counter
from typing import List, Set, Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_rrgen_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, truncate_target=False, append_source_id=False,
    num_buckets=0,
    shuffle=True,
    use_sentiment=None,
    use_category=None,
    use_rating=None,
    use_length=None,
    # ext_senti_dict=None,
    # ext_cate_dict=None,
    # ext_rate_dict=None,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    # A-component data
    # senti_datasets = []
    # cate_datasets = []
    # rate_datasets = []

    # import pdb
    # pdb.set_trace()

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl)

        # breakpoint()

        if tgt_dataset is not None:
            # NOTE using the argument
            # --skip-invalid-size-inputs-valid-test results
            # in differing test sets when comparing
            # models if trained with different
            # --max-source-positions and
            # --max-source-positions arguments. 
            # As an alternative, we can just
            # truncate all source and target datasets to
            # --max-source-positions and --max-target-positions
            
            if truncate_target:
                tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions)

            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        # -----------------
        # load A-Components
        # -----------------
        # import pdb
        # pdb.set_trace()

        if use_sentiment:
            # alpha-systems sentiment annotations
            senti_dataset = np.loadtxt(prefix + use_sentiment)

            # if use_sentiment == 'alpha_sentiment' and dataset_impl == 'raw':
                
            #     if not split_exists(split_k, src, tgt, use_sentiment, data_path) and not split_exists(split_k, tgt, src, use_sentiment, data_path):
            #         raise FileNotFoundError('Ext {} dataset not found: {} ({})'.format(use_sentiment, split, data_path))
                
            #     prefix = os.path.join(
            #         data_path, '{}.{}-{}.'.format(split_k, src, tgt))

                # # alpha-systems sentiment annotations
                # senti_dataset = np.loadtxt(prefix + use_sentiment)

            # elif use_sentiment == 'sentiment':
            #     if not split_exists(split_k, src, tgt, use_sentiment, data_path) and not split_exists(split_k, tgt, src, use_sentiment, data_path):
            #         raise FileNotFoundError(
            #             'Ext {} dataset not found: {} ({})'.format(use_sentiment, split, data_path))
            #     prefix = os.path.join(
            #         data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                
            #     if dataset_impl != 'raw': # hack for handling both 'raw' and binarized datasets
            #         senti_dataset = data_utils.load_indexed_dataset(
            #             prefix + use_sentiment, ext_senti_dict, dataset_impl)
            #     else:
            #         senti_dataset = data_utils.load_and_map_simple_dataset(
            #             prefix + use_sentiment, ext_senti_dict, dataset_impl
            #         )
                
        else:
            senti_dataset = None

        # breakpoint()

        if use_category:
            # numericalised categorical values for
            # domain/category [0, 1]
            cate_dataset = np.loadtxt(prefix + use_category).reshape(-1,1)
            
            # if not split_exists(split_k, src, tgt, use_category, data_path) and not split_exists(split_k, tgt, src, use_category, data_path):
            #     raise FileNotFoundError(
            #         'Ext {} dataset not found: {} ({})'.format(use_category, split, data_path))
            # prefix = os.path.join(
            #     data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            # if dataset_impl != 'raw':
            #     cate_dataset = data_utils.load_indexed_dataset(
            #         prefix + use_category, ext_cate_dict, dataset_impl)
            # else:
            #     cate_dataset = data_utils.load_and_map_simple_dataset(
            #         prefix + use_category, ext_cate_dict, dataset_impl
            #     )
        else:
            cate_dataset = None

        if use_rating:
            # normalised categorical rating values [0, 1]
            rate_dataset = np.loadtxt(prefix + use_rating).reshape(-1,1)
            # if not split_exists(split_k, src, tgt, use_rating, data_path) and not split_exists(split_k, tgt, src, use_rating, data_path):
            #     raise FileNotFoundError(
            #         'Ext {} dataset not found: {} ({})'.format(use_rating, split, data_path))
            # prefix = os.path.join(
            #     data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            # if dataset_impl != 'raw':
            #     rate_dataset = data_utils.load_indexed_dataset(
            #         prefix + use_rating, ext_rate_dict, dataset_impl)
            # else:
            #     rate_dataset = data_utils.load_and_map_simple_dataset(
            #         prefix + use_rating, ext_rate_dict, dataset_impl
            #     )
        else:
            rate_dataset = None

        if use_length:
            # normalised categorical src length values [0, 1]
            length_dataset = np.loadtxt(prefix + use_length).reshape(-1,1)
        else:
            length_dataset = None

        if not combine:
            break


    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(
            tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(
            data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return RRGenDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        ext_senti=senti_dataset,
        ext_cate=cate_dataset,
        ext_rate=rate_dataset,
        ext_len=length_dataset,
        # ext_senti_dict=ext_senti_dict,
        # ext_cate_dict=ext_cate_dict,
        # ext_rate_dict=ext_rate_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
    )


@register_task('rrgen_translation')
class RRGenTranslationTask(LegacyFairseqTask):
    """
    Translation approach for seq2seq review response
    generation (RRGen).

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--truncate-target', action='store_true', default=False,
                            help='truncate target to max-target-positions') # see note above
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')

        # fmt: on

        # -----
        # RRGen
        # -----

        # parser.add_argument('--use-sentiment', action='store_true', default=False,
        #                     help='incorporate sentiment annotations for input review')
        # parser.add_argument('--use-category', action='store_true', default=False,
        #                     help='incorporate category annotations for input review')
        # parser.add_argument('--use-rating', action='store_true', default=False,
        #                     help='incorporate rating annotations for input review')
        # parser.add_argument('--use-length', action='store_true', default=False,
        #                     help='incorporate length attribute for input review')

        parser.add_argument('--use-sentiment', default=None,
                            help='incorporate sentiment annotations for input review: specify filename stem, e.g. `sentiment`.')
        parser.add_argument('--use-category', default=None,
                            help='incorporate category annotations for input review')
        parser.add_argument('--use-rating', default=None,
                            help='incorporate rating annotations for input review')
        parser.add_argument('--use-length', default=None,
                            help='incorporate length attribute for input review')

    # def __init__(self, args, src_dict, tgt_dict, ext_senti_dict, ext_cate_dict, ext_rate_dict):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        # A-component attribute dictionaries
        # self.ext_senti_dict = ext_senti_dict
        # self.ext_cate_dict = ext_cate_dict
        # self.ext_rate_dict = ext_rate_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(
            paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(
            paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(
            args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(
            args.target_lang, len(tgt_dict)))

        # if args.use_sentiment is not None:

        #     # account for single sentiment value input
        #     if args.use_sentiment == 'sentiment':
        #         ext_senti_dict = cls.load_normalisation_dictionary(os.path.join(
        #             paths[0], 'dict.{}.txt'.format(args.use_sentiment)))
        #         logger.info('[{}] dictionary: {} types'.format(args.use_sentiment,
        #                                                     len(ext_senti_dict)))
        #     else:
        #         ext_senti_dict = None

        # else:
        #     ext_senti_dict = None

        # if args.use_category is not None:
        #     ext_cate_dict = cls.load_normalisation_dictionary(os.path.join(
        #         paths[0], 'dict.{}.txt'.format(args.use_category)))
        #     logger.info('[{}] dictionary: {} types'.format(args.use_category,
        #                                                    len(ext_cate_dict)))
        # else:
        #     ext_cate_dict = None

        # if args.use_rating is not None:
        #     ext_rate_dict = cls.load_normalisation_dictionary(os.path.join(
        #         paths[0], 'dict.{}.txt'.format(args.use_rating)))
        #     logger.info('[{}] dictionary: {} types'.format(args.use_rating,
        #                                                    len(ext_rate_dict)))
        # else:
        #     ext_rate_dict = None

        return cls(args, src_dict, tgt_dict)
        # return cls(args, src_dict, tgt_dict, ext_senti_dict, ext_cate_dict, ext_rate_dict)

    @staticmethod
    def build_normalisation_dictionary(filenames: List, dict_path: str, senti: bool = False, cate: bool = False, rate: bool = False):
        """Build a normalisation dictionary instead of a
        regular token-index mapping dictionary.

        All data values are mapped to values in range [0,1]

        Args:

            filenames : list of filenames
            dict_path : path to output dictionary file
        """

        data = Counter()
        str_dtype = False
        for filename in filenames:
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    try:
                        data[int(line)] += 1
                    except ValueError:
                        str_dtype = True
                        data[line] += 1
        # for unknown values
        data_items = sorted(list(data.keys()))

        if str_dtype:
            # add placeholder for unknown unseen categories
            data_items.insert('<unk>', 0)
            le = LabelEncoder()
            norm_items = le.fit_transform(data_items)

            scaler = MinMaxScaler()
            norm_items = scaler.fit_transform(
                norm_items.reshape(-1, 1)).flatten()
        else:
            data_items.insert(0, 0)
            # construct a normalisation object
            # using data_items and then flatten
            # to get a 1D array
            scaler = MinMaxScaler()
            norm_items = scaler.fit_transform(
                np.array(data_items).reshape(-1, 1)).flatten()

        d = dict(zip(data_items, norm_items))

        if isinstance(dict_path, str):
            with open(dict_path, "w", encoding="utf-8") as fd:
                # PathManager.mkdirs(os.path.dirname(dict_path))
                # with PathManager.open(dict_path, "w", encoding="utf-8") as fd:
                #     return self.save(fd)
                for k, v in d.items():
                    print("{} {}".format(k, v), file=fd)

            print(f'Successfully wrote ext dictionary: {dict_path}')

        return d

    @staticmethod
    def load_normalisation_dictionary(dict_path: str):
        """Read in custom normalisation dictionary containing
        mapping from data values to normalised
        representation value between [0,1].

        e.g.

        Hotel 0.5
        Restaurant 1.0
        <unk> 0.0

        Args:
            dict_path : path to normalised dictionary prepared
            during preprocessing step
        """

        with open(dict_path, 'r', encoding='utf8') as f:
            lines = f.readlines()

        d = dict()

        for line in lines:
            try:
                data_val, norm_val = line.rstrip().rsplit(" ", 1)
                # norm_val is expected to be a float between
                # [0,1]
                try:
                    d[int(data_val)] = float(norm_val)
                except:
                    d[data_val] = float(norm_val)

            except:
                raise RuntimeError(f'Failed reading {dict_path}!')

        return d

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test).

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # import pdb
        # pdb.set_trace()

        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_rrgen_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source = self.args.truncate_source,
            truncate_target = self.args.truncate_target,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
            # ext_senti_dict=self.ext_senti_dict,
            # ext_cate_dict=self.ext_cate_dict,
            # ext_rate_dict=self.ext_rate_dict,
            use_sentiment=self.args.use_sentiment,
            use_category=self.args.use_category,
            use_rating = self.args.use_rating,
            use_length = self.args.use_length,
            
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        # TODO: update to match training!
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(
                getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(
                getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args))
        return model
    
    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        """
        Overrides `build_generator` func from
        `fairseq/tasks/fairseq_task.py`
        
        Functionality is the same as the original but
        imports SequenceGenerator class from the 
        task-specific `sequence_generator_rrgen`
        """

        # import pdb;pdb.set_trace()

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        # import from custom sequence_generator script
        from .sequence_generator_rrgen import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(
                self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
