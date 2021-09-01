from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet2.enh.informed_encoder.abs_informed_encoder import AbsInformedEncoder

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.pytorch_backend.tacotron2.encoder import encoder_init

from espnet2.enh.layers.dprnn import DPRNN
from espnet2.enh.layers.dprnn import merge_feature
from espnet2.enh.layers.dprnn import split_feature

class EmbeddingEncoder(AbsInformedEncoder):
    def __init__(self, 
            input_size: int, 
            output_size: int, 
            padding_idx: int = 0,
        ):
        super().__init__()
        self.embed = torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx)
        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    def forward(self, xs_pad, ilens):
        xs_pad =  self.embed(xs_pad)
        return xs_pad, ilens

class RNNEncoder(AbsInformedEncoder):
    def __init__(self, 
            input_size: int,
            output_size: int,
            rnn_type: str = "lstm",
            bidirectional: bool = True,
            num_layers: int = 4,
            hidden_size: int = 320,
            dropout: float = 0.0,
            padding_idx: int = 0,
        ):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        rnn_type = ("b" if bidirectional else "") + rnn_type

        self.embed = torch.nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.enc = torch.nn.ModuleList(
            [
                RNN(
                    hidden_size,
                    num_layers,
                    hidden_size,
                    output_size,
                    dropout,
                    typ=rnn_type,
                )
            ]
        )
        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    def forward(self, xs_pad, ilens, prev_states=None):
        xs_pad = self.embed(xs_pad)
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        xs_pad = xs_pad.masked_fill(make_pad_mask(ilens, xs_pad, 1), 0.0)
        return xs_pad, ilens


class TactronEncoder(AbsInformedEncoder):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS Synthesis by Conditioning WaveNet on Mel
    Spectrogram Predictions`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        input_layer="embed",
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(TactronEncoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        if input_layer == "linear":
            self.embed = torch.nn.Linear(idim, econv_chans)
        elif input_layer == "embed":
            self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = (
                    embed_dim if layer == 0 and input_layer == "embed" else econv_chans
                )
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

        self._input_size = idim

    @property
    def input_size(self):
        return self._input_size

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        xs = self.embed(xs).transpose(1, 2)
        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True, enforce_sorted=False)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        xs = x.unsqueeze(0)
        ilens = torch.tensor([x.size(0)])

        return self.forward(xs, ilens)[0][0]


class DPRNNEncoder(AbsInformedEncoder):
    def __init__(self, 
            input_size: int,
            output_size: int,
            rnn_type: str = "lstm",
            bidirectional: bool = True,
            num_layers: int = 4,
            hidden_size: int = 320,
            dropout: float = 0.0,
            padding_idx: int = 0,
        ):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        rnn_type = ("b" if bidirectional else "") + rnn_type

        self.embed = torch.nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.enc = torch.nn.ModuleList(
            [
                RNN(
                    hidden_size,
                    num_layers,
                    hidden_size,
                    output_size,
                    dropout,
                    typ=rnn_type,
                )
            ]
        )
        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    def forward(self, xs_pad, ilens, prev_states=None):
        xs_pad = self.embed(xs_pad)
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        xs_pad = xs_pad.masked_fill(make_pad_mask(ilens, xs_pad, 1), 0.0)
        return xs_pad, ilens


class DPRNNEncoder(AbsInformedEncoder):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS Synthesis by Conditioning WaveNet on Mel
    Spectrogram Predictions`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        embed_dim: int = 256,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        padding_idx=0,
    ):
        """DPRNN encoder model

        Args:
            input_dim: input feature dimension
            output_size: output feature dimension
            embed_dim: text embedding dimension
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            segment_size: dual-path segment size
            dropout: float, dropout ratio. Default is 0.
        """
        super().__init__()
        # store the hyperparameters
        self.embed = torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx)
        self.segment_size = segment_size

        self.dprnn = DPRNN(
            rnn_type=rnn_type,
            input_size=embed_dim,
            hidden_size=unit,
            output_size=output_size,
            dropout=dropout,
            num_layers=layer,
            bidirectional=bidirectional,
        )

        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        feature = self.embed(xs)

        B, T, N = feature.shape
        feature = feature.transpose(1, 2)  # B, N, T
        segmented, rest = split_feature(
            feature, segment_size=self.segment_size
        )  # B, N, L, K

        processed = self.dprnn(segmented)  # B, N, L, K

        processed = merge_feature(processed, rest)  # B, N, T

        processed = processed.transpose(1, 2)  # B, T, N
        processed = processed.view(B, T, N)

        return processed, ilens
