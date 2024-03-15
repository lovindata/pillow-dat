from typing import Literal

from PIL.Image import Image

from PIL_DAT.dat_light import DATLight


def upscale(
    image: Image, scale: Literal[2, 3, 4], post_processing: bool = True
) -> Image:
    """
    Upscales the provided image using the DATLight model.

    Args:
        image (Image): An instance of PIL Image representing the input image.
        scale (Literal[2, 3, 4]): The scale factor for upscaling.
        post_processing (bool, optional): Whether to apply post-processing to the upscaled image.
            Defaults to True. If set to True, performs median filtering with a kernel size of 3
            to remove small artifacts in the upscaled image. Set to False for artworks to retain
            finer details.

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
    return DATLight(scale).upscale(image, post_processing=post_processing)
