from multiprocessing import Process, Queue

import numpy as np
import torch
from PIL import Image

from basicsr.archs.dat_arch import DAT


def img2tensor_fast(img):
    input = (
        torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        / 255
    )
    return input


def tensor2img_fast(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255
    output = Image.fromarray(tensor.byte().numpy())
    return output


def upscale(image: Image.Image, model: DAT) -> Image.Image:
    tensor = img2tensor_fast(image)
    with torch.no_grad():
        tensor = model(tensor)
    image = tensor2img_fast(tensor)
    return image


def task_upscale(queue: Queue, image: Image.Image, model: DAT) -> None:
    queue.put(upscale(image, model))


if __name__ == "__main__":
    #########################
    # DAT_light_x<?>
    model = DAT(
        upscale=4,
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
    weights = torch.load("./experiments/pretrained_models/DAT_light_x4.pth")

    # DAT_S_x<?>
    # model = DAT(
    #     upscale=4,
    #     in_chans=3,
    #     img_size=64,
    #     img_range=1.0,
    #     depth=[6, 6, 6, 6, 6, 6],
    #     embed_dim=180,
    #     num_heads=[6, 6, 6, 6, 6, 6],
    #     expansion_factor=2,
    #     resi_connection="1conv",
    #     split_size=[8, 16],
    # ).eval()
    # weights = torch.load("./experiments/pretrained_models/DAT_S_x4.pth")

    # DAT_2_x<?>
    # model = DAT(
    #     upscale=4,
    #     in_chans=3,
    #     img_size=64,
    #     img_range=1.0,
    #     depth=[6, 6, 6, 6, 6, 6],
    #     embed_dim=180,
    #     num_heads=[6, 6, 6, 6, 6, 6],
    #     expansion_factor=2,
    #     resi_connection="1conv",
    #     split_size=[8, 32],
    # ).eval()
    # weights = torch.load("./experiments/pretrained_models/DAT_2_x4.pth")

    # DAT_x<?>
    # model = DAT(
    #     upscale=4,
    #     in_chans=3,
    #     img_size=64,
    #     img_range=1.0,
    #     depth=[6, 6, 6, 6, 6, 6],
    #     embed_dim=180,
    #     num_heads=[6, 6, 6, 6, 6, 6],
    #     expansion_factor=4,
    #     resi_connection="1conv",
    #     split_size=[8, 32],
    # ).eval()
    # weights = torch.load("./experiments/pretrained_models/DAT_x4.pth")

    model.load_state_dict(weights["params"])
    print("SUCCESS load!")

    #########################
    queue = Queue()
    image = Image.open("./datasets/single/lumine2.png")
    Process(target=task_upscale, args=(queue, image, model)).start()
    image = queue.get()
    image.show()
