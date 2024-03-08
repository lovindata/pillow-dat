from PIL.Image import open

from PIL_DAT.Image import upscale

lumine_image = open(".github/lumine.png")
lumine_image = upscale(lumine_image, 2)
lumine_image.show()
