from typing import Literal

import torch.nn as nn

from basicsr.archs.dat_arch import DAT
from src.PIL_DAT.dat_model import DATModel


class DATLight(DATModel):
    def __init__(self, pth_path: str, scale: Literal[2, 3, 4]) -> None:
        self._cached_model = DAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            img_range=1.0,
            depth=[18],
            embed_dim=60,
            num_heads=[6],
            expansion_factor=2,
            resi_connection="3conv",
            split_size=[8, 32],
            upsampler="pixelshuffledirect",
        ).eval()
        super().__init__(pth_path, scale)

    @property
    def _model(self) -> nn.Module:
        return self._cached_model
