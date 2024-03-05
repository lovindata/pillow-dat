from typing import Literal

from PIL.Image import Image

from PIL_DAT.dat_light import DATLight


def upscale(image: Image, pth_path: str, scale: Literal[2, 3, 4]) -> Image:
    """
    Upscales the provided image using the specified PyTorch model stored at the given file path.

    Args:
        image (Image): An instance of PIL Image representing the input image.
        pth_path (str): The file path to the PyTorch model.
        scale (Literal[2, 3, 4]): The scale factor for upscaling.

    Returns:
        Image: An instance of PIL Image representing the upscaled image.

    Note:
        This function utilizes the DATLight model exclusively for upscaling. If a different model is desired,
        it should be initialized separately, and the upscale method should be invoked directly from the class instance.

        For improved performance, especially when calling this function multiple times, it is recommended to
        instantiate the DATLight class directly and use its upscale method. This avoids redundant loading of model
        weights, resulting in optimized execution.

    Example:
        >>> input_image = Image.open("input_image.jpg")
        >>> upscaled_image = upscale(input_image, "model.pth", 2)
    """
    return DATLight(pth_path, scale).upscale(image)
