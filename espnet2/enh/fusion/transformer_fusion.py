# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from einops.layers.torch import Rearrange
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, make_pad_mask_2d
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_mult_layer import EncoderMultLayer, ASRDecoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet2.enh.fusion.abs_fusion import AbsFusion
import pdb


class TransformerFusion(AbsFusion):
    """Transformer encoder module.

    Args:
        attention_dim: input dim
        attention_dim: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    """

    def __init__(
        self,
        input_size: int = 256,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        self._input_size = input_size

        self.embed_speech = torch.nn.Sequential(
                torch.nn.Linear(input_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        self.embed_informed = pos_enc_class(attention_dim, positional_dropout_rate)

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderMultLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def input_size(self) -> int:
        return self._input_size

    def forward(
        self,
        embed_speech_pad: torch.Tensor,
        embed_informed_pad: torch.Tensor,
        embed_speech_ilens: torch.Tensor,
        embed_informed_ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            embed_speech_pad: input tensor (B, L1, D)
            embed_informed_pad: input tensor (B, L2, D)
            embed_speech_ilens: audio input length (B)
            embed_informed_ilens: informed input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = ~make_pad_mask_2d(embed_speech_ilens,  embed_informed_ilens).to(embed_speech_pad.device)

        if isinstance(embed_speech_pad, ComplexTensor):
            embed_speech_pad = abs(embed_speech_pad)

        embed_speech_pad = self.embed_speech(embed_speech_pad)
        embed_informed_pad = self.embed_informed(embed_informed_pad)
        embed_speech_pad, _, masks = self.encoders(embed_speech_pad, embed_informed_pad, masks)
        if self.normalize_before:
            embed_speech_pad = self.after_norm(embed_speech_pad)

        return embed_speech_pad, embed_speech_ilens, None


class ASRDecoderFusion(AbsFusion):
    """ASR decoder module.

    Args:
        attention_dim: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    """

    def __init__(
        self,
        attention_dim: int = 256,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        self._attention_dim = attention_dim

        self.embed_speech = torch.nn.Sequential(
                torch.nn.Linear(attention_dim, output_size),
#                 Rearrange('b t c -> b c t'),
#                 torch.nn.Conv1d(output_size, output_size, kernel_size=3, padding=1, bias=False),
#                 Rearrange('b c t -> b t c'),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        self.embed_informed = pos_enc_class(output_size, positional_dropout_rate)

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.encoders = repeat(
            num_blocks,
            lambda lnum: ASRDecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def input_size(self) -> int:
        return self._input_size

    def forward(
        self,
        embed_speech_pad: torch.Tensor,
        embed_informed_pad: torch.Tensor,
        embed_speech_ilens: torch.Tensor,
        embed_informed_ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            embed_speech_pad: input tensor (B, L1, D)
            embed_informed_pad: input tensor (B, L2, D)
            embed_speech_ilens: audio input length (B)
            embed_informed_ilens: informed input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        informed_mask = (~make_pad_mask(embed_informed_ilens)[:, None, :]).to(embed_informed_pad.device)
        speech_mask = (~make_pad_mask(embed_speech_ilens))[:, None, :].to(embed_speech_pad.device)

        embed_speech_pad = self.embed_speech(embed_speech_pad)
        embed_informed_pad = self.embed_informed(embed_informed_pad)

        embed_informed_pad, informed_mask, embed_speech_pad, speech_mask = self.encoders(embed_informed_pad, informed_mask, embed_speech_pad, speech_mask)
        if self.normalize_before:
            embed_speech_pad = self.after_norm(embed_speech_pad)

        return embed_speech_pad, embed_speech_ilens, None

