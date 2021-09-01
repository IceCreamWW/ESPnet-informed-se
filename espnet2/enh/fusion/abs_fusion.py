from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch


class AbsFusion(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        input1_ilens: torch.Tensor,
        input2_ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
