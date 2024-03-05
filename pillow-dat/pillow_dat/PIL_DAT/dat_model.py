from abc import ABC, abstractmethod
from typing import Literal, Tuple

import numpy as np
import PIL.Image as PIL
import torch
import torch.nn as nn
from PIL.Image import BICUBIC, Image
from torch import Tensor


class DATModel(ABC):
    @abstractmethod
    def __init__(self, pth_path: str, scale: Literal[2, 3, 4]) -> None:
        self.scale = scale
        self._model.load_state_dict(
            torch.load(pth_path)["params"]
        )  # Raises error in case of incorrect weights

    def upscale(self, image: Image) -> Image:
        def extract_rgb_and_alpha(image: Image) -> Tuple[Image, Image]:
            rgba_image = image.convert("RGBA")
            return rgba_image.convert("RGB"), rgba_image.split()[-1]

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

        source_mode = image.mode
        rgb, alpha = extract_rgb_and_alpha(image)
        tensor = img2tensor(rgb)
        with torch.no_grad():
            tensor = self._model(tensor)
        rgb = tensor2img(tensor)
        alpha = alpha.resize(rgb.size, BICUBIC)
        rgb.putalpha(alpha)
        image = rgb.convert(source_mode)
        return image

    @property
    @abstractmethod
    def _model(self) -> nn.Module:
        pass
