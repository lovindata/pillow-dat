from PIL.Image import open

from PIL_DAT.Image import upscale

apple_image = open(".github/apple.png")
apple_image = upscale(apple_image, 2)
apple_image.show()
