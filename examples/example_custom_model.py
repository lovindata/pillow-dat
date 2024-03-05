from PIL.Image import open

from pillow_dat.PIL_DAT.dat_s import DATS


def main() -> None:
    lumine_image = open("./lumine.png")
    model = DATS("./DAT_S_x4.pth", 4)  # Instantiate a reusable custom model instance
    lumine_image = model.upscale(lumine_image)
    lumine_image.show()


if __name__ == "__main__":
    main()
