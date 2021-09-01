from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch


class AbsInformedEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def input_size(self):
        pass
