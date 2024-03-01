from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import PIL.Image as PIL
import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor


class DATModel(ABC):
    @abstractmethod
    def __init__(self, pth_path: str, scale: Literal[2, 3, 4]) -> None:
        self.scale = scale
        self._model.load_state_dict(
            torch.load(pth_path)["params"]
        )  # Raises error in case of incorrect weights

    def upscale(self, image: Image) -> Image:
        def img2tensor(image: Image) -> Tensor:
            input = (
                torch.tensor(np.array(image), dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255
            )
            return input

        def tensor2img(tensor: Tensor) -> Image:
            tensor = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255
            output = PIL.fromarray(tensor.byte().numpy())
            return output

        tensor = img2tensor(image)
        with torch.no_grad():
            tensor = self._model(tensor)
        image = tensor2img(tensor)
        return image

    @property
    @abstractmethod
    def _model(self) -> nn.Module:
        pass
