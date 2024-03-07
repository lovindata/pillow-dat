from typing import Literal

from PIL.Image import Image

from PIL_DAT.dat_light import DATLight


def upscale(image: Image, scale: Literal[2, 3, 4]) -> Image:
    """
    Upscales the provided image using the DATLight model.

    Args:
        image (Image): An instance of PIL Image representing the input image.
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
        >>> upscaled_image = upscale(input_image, 2)
    """
    return DATLight(scale).upscale(image)
