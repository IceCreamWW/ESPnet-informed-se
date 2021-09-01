# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import numpy as np
from typeguard import check_argument_types

from einops.layers.torch import Rearrange
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet2.enh.fusion.abs_fusion import AbsFusion
import pdb


class ConcatRNNFusion(AbsFusion):
    """ConcatRNNFusion class.

    Args:
        input_size: The number of expected features in the input
        output_size: The number of output features
        hidden_size: The number of hidden features
        bidirectional: If ``True`` becomes a bidirectional LSTM
        use_projection: Use projection layer or not
        num_layers: Number of recurrent layers
        dropout: dropout probability

    """

    def __init__(
        self,
        input_size: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_layers: int = 4,
        hidden_size: int = 320,
        output_size: int = 320,
        dropout: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        rnn_type = ("b" if bidirectional else "") + rnn_type
        self.enc = torch.nn.ModuleList(
            [
                RNN(
                    input_size,
                    num_layers,
                    hidden_size,
                    output_size,
                    dropout,
                    typ=rnn_type,
                )
            ]
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        embed_speech_pad: torch.Tensor,
        embed_informed_pad: torch.Tensor,
        embed_speech_ilens: torch.Tensor,
        embed_informed_ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        assert embed_speech_pad.shape[1] == embed_informed_pad.shape[1]
        ilens = embed_speech_ilens
        xs_pad = torch.cat((embed_speech_pad, embed_informed_pad), dim=2)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        xs_pad = xs_pad.masked_fill(make_pad_mask(ilens, xs_pad, 1), 0.0)
        return xs_pad, ilens, current_states


class ConcatFusion(AbsFusion):
    """Concat Fusion module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
    ):
        assert check_argument_types()
        super().__init__()

    def forward(
        self,
        embed_speech_pad: torch.Tensor,
        embed_informed_pad: torch.Tensor,
        embed_speech_ilens: torch.Tensor,
        embed_informed_ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Fusion by concatenating

        Args:
            embed_speech_pad: input tensor (B, L1, D)
            embed_informed_pad: input tensor (B, L2, D)
            embed_speech_ilens: audio input length (B)
            embed_informed_ilens: informed input length (B)
            prev_states: Not to be used now.
        Returns:
        """

        assert embed_speech_pad.shape[1] == embed_informed_pad.shape[1]
        ilens = embed_speech_ilens
        xs_pad = torch.cat((embed_speech_pad, embed_informed_pad), dim=2)
        return xs_pad, ilens, None

