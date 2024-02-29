import numpy as np
import torch
from PIL import Image

from basicsr.archs.dat_arch import DAT

#########################

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
print(model)

weights = torch.load("./experiments/pretrained_models/DAT_light_x4.pth")
# print(weights["params"].keys())

model.load_state_dict(weights["params"])
print("SUCCESS load!")

#########################

height = 64
width = 64
print(height, width)

x = torch.randn((1, 3, height, width))
# x = model(x)

print(x.shape)
print("SUCCESS predict!")

#########################


def img2tensor_fast(img):
    input = np.array(img).astype("float32").transpose(2, 0, 1)
    input = torch.from_numpy(input / 255).unsqueeze(0)
    return input


def tensor2img_fast(tensor, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).numpy()
    output = Image.fromarray(output)
    return output


input = Image.open("./datasets/single/lumine2.png")
input_tensor = img2tensor_fast(input)
output_tensor = model(input_tensor)
output = tensor2img_fast(
    output_tensor,
)
output.show()
