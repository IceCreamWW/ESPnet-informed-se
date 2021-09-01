from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions

import pdb

@torch.no_grad()
def plot_attentions(
    model: AbsESPnetModel,
    iterator: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    output_dir: Path,
    ngpu: Optional[int],
) -> None:

    assert check_argument_types()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir = output_dir / "att_ws"
    model.eval()
    for ids, batch in iterator:
        assert isinstance(batch, dict), type(batch)
        assert len(next(iter(batch.values()))) == len(ids), (
            len(next(iter(batch.values()))),
            len(ids),
        )
        batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

        # 1. Forwarding model and gathering all attentions
        #    calculate_all_attentions() uses single gpu only.
        att_dict = calculate_all_attentions(model, batch)

        # 2. Plot attentions: This part is slow due to matplotlib
        for k, att_list in att_dict.items():
            assert len(att_list) == len(ids), (len(att_list), len(ids))
            for id_, att_w in zip(ids, att_list):

                if isinstance(att_w, torch.Tensor):
                    att_w = att_w.detach().cpu().numpy()

                if att_w.ndim == 2:
                    att_w = att_w[None]
                elif att_w.ndim > 3 or att_w.ndim == 1:
                    raise RuntimeError(f"Must be 2 or 3 dimension: {att_w.ndim}")

                w, h = plt.figaspect(1.0 / len(att_w))
                fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                axes = fig.subplots(1, len(att_w))
                if len(att_w) == 1:
                    axes = [axes]

                for ax, aw in zip(axes, att_w):
                    ax.imshow(aw.astype(np.float32), aspect="auto")
                    ax.set_title(f"{k}_{id_}")
                    ax.set_xlabel("Input")
                    ax.set_ylabel("Output")
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                if output_dir is not None:
                    p = output_dir / id_ / f"{k}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(p)
