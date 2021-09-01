import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder
import logging


class ConvEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation """

    def __init__(
        self,
        in_channel: int,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channel, channel, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.stride = stride
        self.kernel_size = kernel_size

        self._output_dim = channel

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        # assert input.dim() == 2, "Currently only support single channle input"
        # logging.info(f"shape of input in conv encoder is {input.shape}")
        if input.dim() == 2:
            input = torch.unsqueeze(input, 1)
        else:
            input = input.transpose(-1, -2)

        feature = self.conv1d(input)
        feature = torch.nn.functional.relu(feature)
        feature = feature.transpose(1, 2)

        flens = (ilens - self.kernel_size) // self.stride + 1

        return feature, flens
