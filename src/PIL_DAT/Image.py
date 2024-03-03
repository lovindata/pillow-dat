from typing import Literal

from PIL.Image import Image

from src.PIL_DAT.dat_light import DATLight


def upscale(image: Image, pth_path: str, scale: Literal[2, 3, 4]) -> Image:
    return DATLight(pth_path, scale).upscale(image)
