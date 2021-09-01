from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.tcn import TemporalConvNet
from espnet2.enh.separator.abs_separator import AbsSeparator

import pdb


class TCNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        ref_channel: int = -1,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
        nonlinear: str = "relu",
        use_mask: bool = True,
        output_dim: int = None,
    ):
        """Temporal Convolution Separator

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            layer: int, number of layers in each stack.
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.ref_channel = ref_channel

        if nonlinear not in ("sigmoid", "relu", "tanh", "null"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.tcn = TemporalConvNet(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            C=num_spk,
            norm_type=norm_type,
            causal=causal,
            mask_nonlinear=nonlinear,
            O = output_dim
        )

        self.use_mask = use_mask

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor, input_to_mask=None
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # if complex spectrum
        if isinstance(input, ComplexTensor):
            feature = abs(input)
        else:
            feature = input
        B, L, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, L

        masks = self.tcn(feature)  # B, num_spk, N, L
        masks = masks.transpose(2, 3)  # B, num_spk, L, N
        masks = masks.unbind(dim=1)  # List[B, L, N]

        if self.use_mask:
            if input_to_mask is None:
                masked = [input * m for m in masks]
            else:
                masked = [input_to_mask * m for m in masks] 
        else:
            masked = [m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
