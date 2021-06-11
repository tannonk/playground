#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

Adaption of RRGen model using Fairseq out-of-the-box LSTM Encoder-Doceder

Date: 18 Aug 2020
Author: Tannon Kew 
Email: kew@cl.uzh.ch

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax, FairseqDropout
from torch import Tensor
from typing import Dict, List, Optional, Tuple

# for rescaling A-component features
import numpy as np
# from sklearn.preprocessing import MinMaxScaler


DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('rrgen_lstm')
class RRGenLSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)

        rrgen_lstm_arch(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        max_source_positions = getattr(
            args, 'max_source_positions', DEFAULT_MAX_SOURCE_POSITIONS)
        max_target_positions = getattr(
            args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            # breakpoint()  # can inspect OOV words here
            # s = set(dictionary.symbols)-set(embed_dict.keys())
            # p s
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError(
                    '--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = RRGenLSTMEncoder(
            dictionary=task.source_dictionary,
            # rnn_type=args.rnn_type,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )

        decoder = RRGenLSTMDecoder(
            dictionary=task.target_dictionary,
            # rnn_type=args.rnn_type,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions,
            residuals=False,
            use_senti=args.use_sentiment,
            use_cate=args.use_category,
            use_rate=args.use_rating,
            use_len=args.use_length,
            tie_ext_features=args.tie_ext_features
        )

        return cls(encoder, decoder)

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        prev_output_tokens: Tensor,
        ext_senti: Optional[Tensor] = None,
        ext_cate: Optional[Tensor] = None,
        ext_rate: Optional[Tensor] = None,
        ext_len: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        # breakpoint()

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            ext_senti=ext_senti,
            ext_cate=ext_cate,
            ext_rate=ext_rate,
            ext_len=ext_len,
        )
        return decoder_out


class RRGenLSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
        self, dictionary, rnn_type='LSTM', embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(
                num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )

        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        enforce_sorted: bool = True,
        ext_senti: Optional[Tensor] = None,
        ext_cate: Optional[Tensor] = None,
        ext_rate: Optional[Tensor] = None,
        ext_len: Optional[Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.

            NOTE
            ext_senti, ext_cate and ext_rate are passed to encoder for simplicity but they are not used in the encoder
        """
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data, enforce_sorted=enforce_sorted
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(
            packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_idx*1.0)
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return tuple((
            x,  # seq_len x batch x hidden
            final_hiddens,  # num_layers x batch x num_directions*hidden
            final_cells,  # num_layers x batch x num_directions*hidden
            encoder_padding_mask,  # seq_len x batch
        ))

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -
                        1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple((
            encoder_out[0].index_select(1, new_order),
            encoder_out[1].index_select(1, new_order),
            encoder_out[2].index_select(1, new_order),
            encoder_out[3].index_select(1, new_order),
        ))

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class RRGenLSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        residuals=False,
        use_senti: bool = False,
        use_cate: bool = False,
        use_rate: bool = False,
        use_len: bool = False,
        tie_ext_features: bool = False
    ):
        super().__init__(dictionary)
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.encoder_output_units = encoder_output_units
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers
        self.tie_ext_features = tie_ext_features
        self.use_senti = use_senti
        self.use_cate = use_cate
        self.use_rate = use_rate
        self.use_len = use_len

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(
                num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        # incorporate external attribute features (A-Component)
        # by concatenating the attribute features onto the
        # encoder outputs. Here we simply increase the size
        # of the hidden vector before it gets passed through
        # MLP

        # import pdb;pdb.set_trace()

        self.tie_ext_feature_size = encoder_output_units

        if self.tie_ext_features:
            if self.use_senti:
                if self.use_senti == 'sentiment':
                    self.tie_ext_feature_size += 1
                elif self.use_senti == 'alpha_sentiment':
                    # for alpha_seystems sentiment, we expect
                    # the sentiment attribute to be a 25d vector
                    self.tie_ext_feature_size += 25
            if self.use_rate:
                self.tie_ext_feature_size += 1
            if self.use_cate:
                self.tie_ext_feature_size += 1
            if self.use_len:
                self.tie_ext_feature_size += 1

        # from original -> handles projections from
        # hidden_size+attribute_feats to hidden_size
        if self.tie_ext_feature_size != hidden_size and self.tie_ext_feature_size != 0:
            self.encoder_hidden_proj = Linear(
                self.tie_ext_feature_size, hidden_size)
            self.encoder_cell_proj = Linear(
                self.tie_ext_feature_size, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        # initialise attention
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(
                hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(
                out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
        ext_senti: Optional[Tensor] = None,
        ext_cate: Optional[Tensor] = None,
        ext_rate: Optional[Tensor] = None,
        ext_len: Optional[Tensor] = None,
    ):
        """
        ext_senti: list of tensor values for input reviews
        ext_cate: list of tensor values for input reviews
        ext_rate: list of tensor values for input reviews
        ext_len: list of tensor values for input reviews
        """

        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state, ext_senti, ext_cate, ext_rate, ext_len
        )
        return self.output_layer(x), attn_scores

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        # A-Component
        ext_senti: Optional[Tensor] = None,
        ext_cate: Optional[Tensor] = None,
        ext_rate: Optional[Tensor] = None,
        ext_len: Optional[Tensor] = None,
    ):
        """

        Similar to *forward* but only return features.

        Modified to accept A-Components

        """
        # get outputs from encoder
        if encoder_out is not None:
            # torch.Size([max_src_len, bsz, hidden_size])
            encoder_outs = encoder_out[0]
            # torch.Size([1, bsz, hidden_size])
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]  # torch.Size([1, bsz, hidden_size])
            # torch.Size([max_src_len, bsz])
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)

        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        # bsz, tgt_len, vec_size -> tgt_len, bsz, vec_size
        x = x.transpose(0, 1)

        # import pdb
        # pdb.set_trace()

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state)
        elif encoder_out is not None:
            # setup recurrent cells

            if self.use_senti and ext_senti is not None:
                encoder_hiddens, encoder_cells = self.extend_hidden_state(
                    encoder_hiddens, encoder_cells, ext_senti)
            if self.use_cate and ext_cate is not None:
                encoder_hiddens, encoder_cells = self.extend_hidden_state(
                    encoder_hiddens, encoder_cells, ext_cate)
            if self.use_rate and ext_rate is not None:
                encoder_hiddens, encoder_cells = self.extend_hidden_state(
                    encoder_hiddens, encoder_cells, ext_rate)
            if self.use_len and ext_len is not None:
                encoder_hiddens, encoder_cells = self.extend_hidden_state(
                    encoder_hiddens, encoder_cells, ext_len)

            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(
                    y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        # attention
        assert srclen > 0 or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(
            srclen, seqlen, bsz) if self.attention is not None else None
        outs = []

        # seqlen = tgt output len
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell

                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None

                # NOTE
                # hidden = Tensor(bsz, hidden_size)
                # encoder_outs = Tensor(srclen, bsz, hidden_size)
                # encoder_padding_mask = Tensor(srclen, bsz)
                out, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask)

            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(
            incremental_state, 'cached_state', cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(
            incremental_state, 'cached_state')
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        # can be None for decoder-only language models
        input_feed = cached_state["input_feed"]
        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(
            incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(
            incremental_state, 'cached_state', cached_state_new),
        return

    def reorder_ext_attributes(self, ext_atts: Tensor, new_order: Tensor):
        """
        Reorder an A-component attribute vector for decoding
        with beam search

        ext_vector : e.g. ext_senti Tensor([batch_size]), etc.
        new_order : index Tensor indicating how to reorder
        the input tensor
        """
        # import pdb
        # pdb.set_trace()

        # for naive approach where ext are 1d tensors e.g.
        # [1.0, 1.0, 0.0, 1.0]
        # .view(-1) => flattens tensor e.g. [4, 25] => [100]
        # return ext_atts.view(1, -1).index_select(1, new_order).view(-1)
        
        return ext_atts.index_select(0, new_order)


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    @staticmethod
    def extend_hidden_state(encoder_hiddens: Tensor, encoder_cells: Tensor, ext_atts: Tensor):
        """
        Concatenate ext attribute vectors to current hidden
        states and cells along axis 1

        Args:
            encoder_hiddens : final encoder hidden states
            encoder_cells : final encoder hidden cells
            ext_vector : attribute features, e.g. sentiment,
            category, rating
        """

        # import pdb
        # pdb.set_trace()

        # repeat ext_senti attribute vector by
        # number of hidden states in encoder output
        ext_atts = ext_atts.repeat(len(encoder_hiddens), 1)

        # concatenate attribute features with
        # encoder hidden states
        # encoder_hiddens = [torch.cat((h, v.unsqueeze(-1)), 1) for h, v in zip(encoder_hiddens, ext_atts)]
        # encoder_hiddens = [torch.cat((h, ext_atts), 1) for h in encoder_hiddens]
        new_hiddens = []
        for hidden in encoder_hiddens:
            new_hiddens.append(torch.cat((hidden, ext_atts), 1))

        # encoder_cells = [torch.cat((c, v.unsqueeze(-1)), 1) for c, v in zip(encoder_cells, ext_atts)]
        # encoder_cells = [torch.cat((c, ext_atts), 1) for h in encoder_cells]
        new_cells = []
        for cell in encoder_cells:
            new_cells.append(torch.cat((cell, ext_atts), 1))
        
        return new_hiddens, new_cells


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


# -----
# RRGen
# -----

# The first argument to ``@register_model_architecture()`` should be the name
# of the model. The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture
@register_model_architecture('rrgen_lstm', 'rrgen_lstm_arch')
def rrgen_lstm_arch(args):
    args.rnn_type = getattr(args, 'rnn_type', 'LSTM')  # TODO extend for GRU
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 100)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(
        args, 'encoder_hidden_size', 200)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(
        args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 100)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(
        args, 'decoder_hidden_size', 200)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(
        args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(
        args, 'adaptive_softmax_cutoff', '10000,50000,200000')
    args.learning_rate = getattr(args, 'learning_rate', 0.001)
    args.skip_invalid_size_inputs_valid_test = getattr(
        args, 'skip_invalid_size_inputs_valid_test', True)
    args.max_tokens = getattr(args, 'max_tokens', 2000)
    args.batch_size = getattr(args, 'batch_size', 8)
    # args.num_workers = getattr(args, 'num-workers', 0) # good for debugging
    args.tie_ext_features = getattr(args, 'tie_ext_features', True)
