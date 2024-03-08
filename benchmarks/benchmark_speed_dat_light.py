import gc
from time import time
from typing import List, Literal

from PIL.Image import Image, new

from PIL_DAT.dat_light import DATLight


def print_exec_time(model: DATLight, image: Image) -> None:
    gc.collect()
    start_time = time()
    model.upscale(image)
    execution_time = time() - start_time
    print(
        f"DAT light (x{model.scale}) and image {image.size}:",
        round(execution_time, 1),
        "seconds",
    )


scales: List[Literal[2, 3, 4]] = [2, 3, 4]
image_sizes = [320, 640, 960, 1280, 1920]
for scale in scales:
    model = DATLight(scale)
    for image_size in image_sizes:
        if (scale * image_size) <= 2560:
            image = new("RGBA", (image_size, image_size))
            print_exec_time(model, image)
