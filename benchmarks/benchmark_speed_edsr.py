import gc
from time import time
from typing import List, Literal

import cv2
import numpy as np
from cv2 import dnn_superres
from cv2.dnn_superres import DnnSuperResImpl
from PIL.Image import new


def print_exec_time(
    model: DnnSuperResImpl, model_scale: Literal[2, 3, 4], image: np.ndarray
) -> None:
    gc.collect()
    start_time = time()
    model.upsample(image)
    execution_time = time() - start_time
    print(
        f"EDSR (x{model_scale}) and image {image.shape[:2]}:",
        round(execution_time, 1),
        "seconds",
    )


scales: List[Literal[2, 3, 4]] = [2, 3, 4]
image_sizes = [320, 640, 960, 1280, 1920]
for scale in scales:
    model = dnn_superres.DnnSuperResImpl.create()
    model.setModel("edsr", scale)
    model.readModel(f"./dist/EDSR_x{scale}.pb")
    for image_size in image_sizes:
        if (scale * image_size) <= 2560:
            image = new("RGBA", (image_size, image_size))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            print_exec_time(model, scale, image)
