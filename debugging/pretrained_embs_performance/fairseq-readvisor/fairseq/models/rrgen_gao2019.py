# #!/usr/bin/python3
# # -*- coding: utf-8 -*-

"""


NB. in progress...

TODO:
Adapt encoder, i.e. make compatible with fairseq
Adapt decoder
Create model class and register

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax, FairseqDropout
from torch import Tensor
from typing import Dict, List, Optional, Tuple


DEFAULT_MAX_SOURCE_POSITIONS = 400
DEFAULT_MAX_TARGET_POSITIONS = 400


def sequence_mask(sequence_length, max_len=None):
    """
    Used by Decoder
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand

    return mask


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

# class EncoderRNN(nn.Module):


class EncoderRNN(FairseqEncoder):
    """

    Adapted Encoder from Gao et al. 2019 

    https://gitlab.uzh.ch/rereadvisor/gao2019_rrgen/-/blob/master/src/encoder.py

    """

    def __init__(self, dictionary, embed_dim=100, rnn_type='GRU', hidden_size=128, num_layers=1, dropout=0.3, bidirectional=False, pretrained_embed=None, padding_idx=None, dropout_in=0.1, dropout_out=0.1, max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS):
        """

        Changes: 
        replaced 'embedding=None' with dictionary,
        embed_dim=100
        added: 
            `dropout_in=0.1`, `dropout_out=0.1` (required by FairseqEncoder)
            `padding_idx`

        """

        # super(EncoderRNN, self).__init__(dictionary)
        super().__init__(dictionary)

        self.num_layers = num_layers
        self.dropout = dropout_in
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions

        # self.embedding = embedding
        # self.word_vec_size = self.embedding.embedding_dim

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(
                num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.rnn_type = rnn_type

        self.rnn = getattr(nn, self.rnn_type)(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)

    def forward(self, src_seqs: Tensor, src_lengths: Tensor, hidden=None):
        """Encoder Forward Pass

        Args:
            - src_seqs: (max_src_len, batch_size)
            - src_lengths: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """

        # import pdb
        # pdb.set_trace()

        # (max_src_len, batch_size) => (max_src_len, batch_size, word_vec_size)
        emb = self.embed_tokens(src_seqs)

        # need to transpose output (see fairseq LSTM model)
        emb = emb.transpose(0, 1)

        # packed_emb:
        # - data: (sum(batch_sizes), word_vec_size)
        # - batch_sizes: list of batch sizes
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, src_lengths)

        # import pdb
        # pdb.set_trace()

        # rnn(gru) returns:
        # - packed_outputs: shape same as packed_emb
        # - hidden: (num_layers * num_directions, batch_size, hidden_size)
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        # output_lens == src_lengths
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        return outputs, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


# class LuongAttnDecoderRNN(nn.Module):
class LuongAttnDecoderRNN(FairseqIncrementalDecoder):

    def __init__(self,
                 dictionary,
                 encoder,
                 embed_dim=512,
                 hidden_size=512,
                 out_embed_dim=512,
                 num_layers=1,
                 dropout_in=0.1,
                 dropout_out=0.1,
                 attention=True,
                 encoder_output_units=512,
                 pretrained_embed=None,
                 share_input_output_embed=True,
                 adaptive_softmax_cutoff=None,
                 max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
                 embedding=None,
                 bias=True,
                 dropout=0.3,
                 tie_ext_feature=False,
                 ext_senti=None,
                 ext_rate=None,
                 ext_cate=None,
                 ext_seqlen=None):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025

            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462

        Changes:

            replace `tie_embeddings` with `share_input_output_embed`

        """

        # super(LuongAttnDecoderRNN, self).__init__()
        super().__init__(dictionary)

        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        self.embedding = embedding
        self.attention = attention
        self.share_input_output_embed = share_input_output_embed
        self.tie_ext_feature = tie_ext_feature
        self.use_rate = use_rate
        self.use_cate = use_cate
        self.use_seqlen = use_seqlen
        self.use_senti = use_senti

        # rate_size

        #     self.ext_rate_size = self.ext_rate_embedding.embedding_dim
        # else:
        #     self.ext_rate_size = 0
        # # appcate_size
        # if self.ext_appcate_embedding:
        #     self.ext_appcate_size = self.ext_appcate_embedding.embedding_dim
        # else:
        #     self.ext_appcate_size = 0
        # # seq_len
        # if self.ext_seqlen_embedding:
        #     self.ext_seqlen_size = self.ext_seqlen_embedding.embedding_dim
        # else:
        #     self.ext_seqlen_size = 0
        # # sentiment
        self.ext_senti_size = 1 if self.use_senti else 0

        self.vocab_size = len(dictionary)
        # self.vocab_size = self.embedding.num_embeddings
        self.word_vec_size = 100  # TODO infer self.embedding.embedding_dim

        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(
                self.vocab_size, self.word_vec_size, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.word_vec_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout)

        if self.attention:
            self.W_a = nn.Linear(encoder.hidden_size * encoder.num_directions,
                                 self.hidden_size, bias=bias)
            self.W_c = nn.Linear(encoder.hidden_size * encoder.num_directions + self.hidden_size,
                                 self.hidden_size, bias=bias)

        if self.share_input_output_embed:
            self.W_proj = nn.Linear(
                self.hidden_size, self.word_vec_size, bias=bias)
            self.W_s = nn.Linear(self.word_vec_size,
                                 self.vocab_size, bias=bias)
            self.W_s.weight = self.embedding.weight
        else:
            self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)

        input_size = self.hidden_size

        # import pdb
        # pdb.set_trace()

        if self.share_input_output_embed:
            if self.ext_rate_embedding:
                input_size += self.ext_rate_size
            if self.ext_senti_embedding:
                input_size += self.ext_senti_size
            if self.ext_appcate_embedding:
                input_size += self.ext_appcate_size
            if self.ext_seqlen_embedding:
                input_size += self.ext_seqlen_size

        self.W_r = nn.Linear(input_size, self.hidden_size, bias=bias)
        # self.ext_rate_size+self.hidden_size+self.ext_appcate_size+self.ext_seqlen_size+self.ext_senti_size

    def forward(self, encoder_out, src_lengths, prev_output_tokens=None, incremental_state=None, rate=None, cate=None, ext_senti=None):
        """Decoder Forward Pass

        Args:

            - encoder_outputs: Tensor(max_src_len, batch_size, hidden_size * num_directions)
            - src_lens : Tensor(batch_size)
            - prev_output_tokens : Tensor(batch_size, longest_target_seq_len) -> target outputs
            - rate_sents : Tensor(batch_size)
            - cate_sents : Tensor(batch_size)
            - srclen_cates : Tensor(batch_size)
            - senti_sents : Tensor(batch_size)

            # encoder_out = (outputs, hidden)
            # encoder_out = (Tensor, (Tensor, Tensor))
            # encoder_out[0].size()
            # torch.Size([414, 8, 200])
            # encoder_out[1][0].size()
            # torch.Size([1, 8, 200])
            # encoder_out[1][1].size()
            # torch.Size([1, 8, 200])

        changes:

        replaced: 
        input_seq => prev_output_tokens (should be
        initialised with BOS symbol)
        - input_seq      : Tensor(batch_size)

        decoder_hidden, encoder_outputs => encoder_out
        - decoder_hidden : (t = 0) last encoder hidden state 
                                Tensor(num_layers * num_directions, batch_size, hidden_size) 
                            (t > 0) previous decoder hidden state 
                                Tensor(num_layers, batch_size, hidden_size)


        """

        # import pdb
        # pdb.set_trace()

        if incremental_state is not None:
            # If the *incremental_state* argument is not ``None`` then we are
            # in incremental inference mode. While *prev_output_tokens* will
            # still contain the entire decoded prefix, we will only use the
            # last step and assume that the rest of the state is cached.
            prev_output_tokens = prev_output_tokens[:, -1:]

            # TODO: ensure initialised prev_output_tokens as ~
            # torch.LongTensor([BOS] * batch_size)
            # for first forward!

        bsz, tgt_len = prev_output_tokens.size()

        encoder_outputs, encoder_hidden = encoder_out
        # LSTM hidden state is a tuple (hidden_state, cell_state)
        if isinstance(encoder_hidden, tuple):
            final_encoder_hidden, final_encoder_cell = encoder_hidden
        else:  # GRU hidden state is not
            final_encoder_hidden = encoder_hidden

        # hidden_size =
        # Tensor(num_layers (?), bsz, hidden_size)
        #

        # (batch_size) => (seq_len=1, batch_size)
        # prev_output_tokens = prev_output_tokens.unsqueeze(0)
        # prev_output_tokens = prev_output_tokens[:, -1:].transpose(0, 1)

        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size)
        emb = self.embed_tokens(prev_output_tokens)

        # TODO Apply dropout.
        # emb = self.dropout(emb)

        # simple_lstm.py: Concatenate the Encoder's final hidden state to *every* embedded
        # target token for teacher forcing
        # NOTE:
        # (Pdb) final_encoder_hidden => Tensor([1, 8, 200])
        # (Pdb) final_encoder_hidden.transpose(0, 1) => Tensor([8, 1, 200])
        # (Pdb) final_encoder_hidden.transpose(0,
        # 1).expand(bsz, tgt_len, -1) => Tensor([8, 309, 200])
        # emb = torch.cat([emb, final_encoder_hidden.transpose(
        #     0, 1).expand(bsz, tgt_len, -1)], dim=2)
        # emb = Tensor(bsz, tgt_length, word_vec_size+hidden_size)
        # e.g. Tensor([8, 309, 300])

        # Add external embeddings: (batch_size, feature_size) => (num_layers, batch_size, feature_size
        # if self.ext_rate_embedding:
        #     ext_rate_embedding = self.ext_rate_embedding(rate_sents)
        #     ext_rate_embedding = ext_rate_embedding.unsqueeze(
        #         0).repeat(self.num_layers, 1, 1)
        # if self.ext_appcate_embedding:
        #     ext_appcate_embedding = self.ext_appcate_embedding(cate_sents)
        #     ext_appcate_embedding = ext_appcate_embedding.unsqueeze(
        #         0).repeat(self.num_layers, 1, 1)
        # if self.ext_seqlen_embedding:
        #     ext_seqlen_embedding = self.ext_seqlen_embedding(srclen_cates)
        #     ext_seqlen_embedding = ext_seqlen_embedding.unsqueeze(
        #         0).repeat(self.num_layers, 1, 1)
        # if self.ext_senti:
        #     # ext_senti_embedding = self.ext_senti_embedding(senti_sents)
        #     ext_senti_embedding = ext_senti_embedding.unsqueeze(
        #         0).repeat(self.num_layers, 1, 1)

        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)

        # if self.tie_ext_feature:
        #     if self.ext_rate_embedding:
        #         decoder_hidden = torch.cat(
        #             (decoder_hidden, ext_rate_embedding), 2)
        #     if self.ext_appcate_embedding:
        #         decoder_hidden = torch.cat(
        #             (decoder_hidden, ext_appcate_embedding), 2)
        #     if self.ext_seqlen_embedding:
        #         decoder_hidden = torch.cat(
        #             (decoder_hidden, ext_seqlen_embedding), 2)
        #     if self.ext_senti_embedding:
        #         decoder_hidden = torch.cat(
        #             (decoder_hidden, ext_senti_embedding), 2)
        # decoder_hidden = torch.cat((decoder_hidden, ext_rate_embedding, ext_appcate_embedding, ext_seqlen_embedding, ext_senti_embedding), 2)

        # NOTE: this transposition was causing error
        # final_encoder_hidden = torch.tanh(self.W_r(final_encoder_hidden))

        # We will now check the cache and load the cached previous hidden and
        # cell states, if they exist, otherwise we will initialize them to
        # zeros (as before). We will use the ``utils.get_incremental_state()``
        # and ``utils.set_incremental_state()`` helpers.
        initial_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )
        if initial_state is None:
            # first time initialization, same as the original version
            # initial_state = (
            #     final_encoder_hidden,
            #     torch.zeros_like(final_encoder_hidden)
            #     # final_encoder_hidden.unsqueeze(0),  # hidden
            #     # torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
            # )
            # initial_state = final_encoder_hidden
            initial_state = torch.zeros_like(final_encoder_hidden)

        decoder_output, decoder_hidden = self.rnn(
            emb.transpose(0, 1),  # convert to shape `(tgt_len, bsz, dim)`
            initial_state
        )

        # decoder_output = Tensor([309, bsz, hidden_size])

        import pdb
        pdb.set_trace()

        # Update the cache with the latest hidden and cell states.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', decoder_hidden,
        )

        # (seq_len=1, batch_size, hidden_size) => (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0, 1)

        """ 
        ------------------------------------------------------------------------------------------
        Notes of computing attention scores
        ------------------------------------------------------------------------------------------
        # For-loop version:

        max_src_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attention_scores = Variable(torch.zeros(batch_size, max_src_len))

        # For every batch, every time step of encoder's hidden state, calculate attention score.
        for b in range(batch_size):
            for t in range(max_src_len):
                # Loung. eq(8) -- general form content-based attention:
                attention_scores[b,t] = decoder_output[b].dot(attention.W_a(encoder_outputs[t,b]))

        ------------------------------------------------------------------------------------------
        # Vectorized version:

        1. decoder_output: (batch_size, seq_len=1, hidden_size)
        2. encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        3. W_a(encoder_outputs): (max_src_len, batch_size, hidden_size)
                        .transpose(0,1)  : (batch_size, max_src_len, hidden_size) 
                        .transpose(1,2)  : (batch_size, hidden_size, max_src_len)
        4. attention_scores: 
                        (batch_size, seq_len=1, hidden_size) * (batch_size, hidden_size, max_src_len) 
                        => (batch_size, seq_len=1, max_src_len)
        """

        if self.attention:

            # attention_scores: (batch_size, seq_len=1, max_src_len)
            attention_scores = torch.bmm(decoder_output, self.W_a(
                encoder_outputs).transpose(0, 1).transpose(1, 2))

            # attention_mask: (batch_size, seq_len=1, max_src_len)
            attention_mask = sequence_mask(src_lengths).unsqueeze(1)

            # Fills elements of tensor with `-float('inf')` where `mask` is False.
            attention_scores.data.masked_fill_(
                ~attention_mask.data, -float('inf'))

            # print('ATT SCORES:')
            # print(attention_scores)

            # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax`
            # => (batch_size, seq_len=1, max_src_len)
            # try:  # torch 0.3.x
            # attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)
            # except:
            attention_weights = F.softmax(
                attention_scores.squeeze(1)).unsqueeze(1)

            # context_vector:
            # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
            # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
            context_vector = torch.bmm(
                attention_weights, encoder_outputs.transpose(0, 1))

            # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
            concat_input = torch.cat([context_vector, decoder_output], -1)

            # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)
            # UPGRADE
            # concat_output = F.tanh(self.W_c(concat_input))
            concat_output = torch.tanh(self.W_c(concat_input))

            # Prepare returns:
            # (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len)
            attention_weights = attention_weights.squeeze(1)
        else:
            attention_weights = None
            concat_output = decoder_output

        # If input and output embeddings are tied,
        # project `decoder_hidden_size` to `word_vec_size`.
        if self.share_input_output_embed:
            output = self.W_s(self.W_proj(concat_output))
        else:
            # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)
            output = self.W_s(concat_output)

            # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)

        del src_lengths

        import pdb
        pdb.set_trace()

        return output, decoder_hidden, attention_weights


@register_model('gao_rrgen_lstm')
class LSTMModel(FairseqEncoderDecoderModel):
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

        # import pdb
        # pdb.set_trace()

        gao_rrgen_lstm_arch(args)

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

        # import pdb;pdb.set_trace()

        encoder = EncoderRNN(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )

        decoder = LuongAttnDecoderRNN(
            dictionary=task.target_dictionary,
            encoder=encoder,
            # embed_dim=args.decoder_embed_dim,
            # hidden_size=args.decoder_hidden_size,
            # out_embed_dim=args.decoder_out_embed_dim,
            # num_layers=args.decoder_layers,
            # dropout_in=args.decoder_dropout_in,
            # dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            # encoder_output_units=encoder.output_units,
            # pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            # adaptive_softmax_cutoff=(
            #     options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
            #     if args.criterion == 'adaptive_loss' else None
            # ),
            # max_target_positions=max_target_positions,
            # residuals=False,
            # ext_senti=args.use_sentiment,
            # ext_cate=args.use_category,
            # ext_rate=args.use_rating
        )

        return cls(encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens=None,
        ext_senti=None,
        cate=None,
        rate=None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        # import pdb
        # pdb.set_trace()

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        # encoder_out = (outputs, hidden)
        # if LSTM:
        # encoder_out = (Tensor, (Tensor, Tensor))
        # encoder_out[0].size()
        # torch.Size([414, 8, 200])
        # encoder_out[1][0].size()
        # torch.Size([1, 8, 200])
        # encoder_out[1][1].size()
        # torch.Size([1, 8, 200])
        # if GRU:
        # encoder_out = (Tensor, Tensor)

        # import pdb
        # pdb.set_trace()

        decoder_out = self.decoder(
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            prev_output_tokens=prev_output_tokens,
            # incremental_state=incremental_state,
            use_senti=use_senti,
            use_cate=use_cate,
            use_rate=use_rate
        )
        return decoder_out


@register_model_architecture('gao_rrgen_lstm', 'gao_rrgen_lstm_arch')
def gao_rrgen_lstm_arch(args):
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
    args.learning_rate = getattr(args, 'learning-rate', 0.005)
    args.optimizer = getattr(args, 'optimizer', 'adam')
    args.skip_invalid_size_inputs_valid_test = getattr(
        args, 'skip-invalid-size-inputs-valid-test', True)
    args.max_tokens = getattr(args, 'max-tokens', 4000)
    args.batch_size = getattr(args, 'batch-size', 8)
    args.num_workers = getattr(args, 'num-workers', 0)
