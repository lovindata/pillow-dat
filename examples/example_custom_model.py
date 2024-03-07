from PIL.Image import open

from PIL_DAT.dat_base import DATBase


def main() -> None:
    lumine_image = open(".github/lumine.png")
    model = DATBase(
        "./dist/DAT_x4.pth", 4
    )  # Instantiate a reusable custom model instance
    lumine_image = model.upscale(lumine_image)
    lumine_image.show()


if __name__ == "__main__":
    main()
