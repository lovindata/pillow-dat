from typing import Literal

import torch.nn as nn

from PIL_DAT._dat_arch import DAT
from PIL_DAT.dat_model import DATModel


class DATBase(DATModel):
    def __init__(self, pth_path: str, scale: Literal[2, 3, 4]) -> None:
        self._cached_model = DAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            img_range=1.0,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            expansion_factor=4,
            resi_connection="1conv",
            split_size=[8, 32],
        ).eval()
        super().__init__(pth_path, scale)

    @property
    def _model(self) -> nn.Module:
        return self._cached_model
