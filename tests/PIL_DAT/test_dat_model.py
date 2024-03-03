from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch.nn as nn
from PIL import Image

from src.PIL_DAT.dat_light import DATLight
from src.PIL_DAT.dat_model import DATModel


class TestDATModel:
    @pytest.mark.parametrize("scale", [(2), (3), (4)])
    def test_upscale_scale(self, scale: Literal[2, 3, 4]) -> None:
        input = Image.new("RGB", (4, 4))
        with patch("torch.load", return_value={"params": Mock()}), patch.object(
            nn.Module, "load_state_dict", return_value=None
        ):
            model: DATModel = DATLight(Mock(), scale)
            output = model.upscale(input)
            assert output.size[0] == (input.size[0] * scale)
            assert output.size[1] == (input.size[1] * scale)

    @pytest.mark.parametrize(
        "image_mode",
        [
            "1",  # 1-bit pixels, black and white, stored with one pixel per byte
            "L",  # 8-bit pixels, grayscale
            "P",  # 8-bit pixels, mapped to any other mode using a color palette
            "RGB",  # 3x8-bit pixels, true color
            "RGBA",  # 4x8-bit pixels, true color with transparency mask
            "CMYK",  # 4x8-bit pixels, color separation
            "YCbCr",  # 3x8-bit pixels, color video format
            # "LAB",  # 3x8-bit pixels, the L*a*b color space (unexpected dependency issue, https://github.com/python-pillow/Pillow/issues/5308)
            "HSV",  # 3x8-bit pixels, Hue, Saturation, Value color space
            "I",  # 32-bit signed integer pixels
            "F",  # 32-bit floating point pixels
            "LA",  # L with alpha
            "PA",  # P with alpha
            "RGBX",  # true color with padding
            "RGBa",  # true color with premultiplied alpha
        ],
    )
    def test_upscale_image_mode_supported(self, image_mode: str) -> None:
        input = Image.new(image_mode, (4, 4))
        with patch("torch.load", return_value={"params": Mock()}), patch.object(
            nn.Module, "load_state_dict", return_value=None
        ):
            model: DATModel = DATLight(Mock(), 2)
            output = model.upscale(input)
            assert output.mode == image_mode

    @pytest.mark.parametrize(
        ("image_mode", "expected"),
        [
            (
                "La",
                ValueError("conversion from La to L not supported"),
            ),  # L with premultiplied alpha
            (
                "I;16",
                ValueError("conversion from RGB to I;16 not supported"),
            ),  # 16-bit unsigned integer pixels
            (
                "I;16L",
                ValueError("conversion from RGB to I;16L not supported"),
            ),  # 16-bit little endian unsigned integer pixels
            (
                "I;16B",
                ValueError("conversion from RGB to I;16B not supported"),
            ),  # 16-bit big endian unsigned integer pixels
            (
                "I;16N",
                ValueError("conversion from RGB to I;16N not supported"),
            ),  # 16-bit native endian unsigned integer pixels
            (
                "BGR;15",
                ValueError("conversion from BGR;15 to RGB not supported"),
            ),  # 15-bit reversed true colour
            (
                "BGR;16",
                ValueError("conversion from BGR;16 to RGB not supported"),
            ),  # 16-bit reversed true colour
            (
                "BGR;24",
                ValueError("conversion from BGR;24 to RGB not supported"),
            ),  # 24-bit reversed true colour
        ],
    )
    def test_upscale_image_mode_unsupported(
        self, image_mode: str, expected: Exception
    ) -> None:
        input = Image.new(image_mode, (4, 4))
        with patch("torch.load", return_value={"params": Mock()}), patch.object(
            nn.Module, "load_state_dict", return_value=None
        ):
            model: DATModel = DATLight(Mock(), 2)
            with pytest.raises(type(expected)) as output:
                model.upscale(input)
            assert str(output.value) == str(expected)
