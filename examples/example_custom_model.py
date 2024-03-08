from PIL.Image import open

from PIL_DAT.dat_s import DATS

lumine_image = open(".github/lumine.png")
model = DATS("./dist/DAT_S_x4.pth", 4)  # Instantiate a reusable custom model instance
lumine_image = model.upscale(lumine_image)
lumine_image.show()
