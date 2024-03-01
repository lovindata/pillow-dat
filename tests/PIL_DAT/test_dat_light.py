from typing import Literal

import numpy as np
import pytest
from PIL import Image

from src.PIL_DAT.dat_light import DATLight


class TestDATLight:
    @pytest.mark.parametrize("scale", [(2), (3), (4)])
    def test_upscale(self, scale: Literal[2, 3, 4]) -> None:
        input = Image.fromarray(
            np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
        )
        model = DATLight("", scale)
        output = model.upscale(input)
        assert output.size[0] == (input.size[0] * scale)
        assert output.size[1] == (input.size[1] * scale)
