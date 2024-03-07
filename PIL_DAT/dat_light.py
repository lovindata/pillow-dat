from importlib.resources import files
from typing import Literal

import torch.nn as nn

from PIL_DAT import resources
from PIL_DAT._dat_arch import DAT
from PIL_DAT.dat_model import DATModel


class DATLight(DATModel):
    def __init__(self, scale: Literal[2, 3, 4]) -> None:
        """
        Initializes an instance of DATLight, a lightweight and fast version of the DAT model for image upscaling.

        This class inherits from DATModel and provides a simplified interface for quickly upscaling images using a pre-embarked DATLight model included in the package.

        Args:
            scale (Literal[2, 3, 4]): The scaling factor to be used for upscaling images. It must be one of the following: 2, 3, or 4.

        Note:
            DATLight utilizes a pre-embarked DATLight model, which is optimized for faster inference and reduced memory footprint.
            The pre-embarked model is internally loaded and used for upscaling images, eliminating the need to provide a path to the model weights file during initialization.
            Make sure to select a valid scaling factor supported by the embarked model (2, 3, or 4).
        """
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
        pth_path = str(
            files(resources) / f"DAT_light_x{scale}.pth"
        )  # Embarked model path
        super().__init__(pth_path, scale)

    @property
    def _model(self) -> nn.Module:
        return self._cached_model
