from PIL.Image import open

from src.PIL_DAT.Image import upscale


def main() -> None:
    lumine_image = open("./lumine.png")
    lumine_image = upscale(lumine_image, "./DAT_light_x2.pth", 2)
    lumine_image.show()


if __name__ == "__main__":
    main()
