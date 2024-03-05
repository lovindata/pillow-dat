<p align="center">
  <a href="https://ilovedatajjia.github.io/blog/" target="_blank">
      <img alt="iLoveData" src="https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/logo.png?raw=true" width="500" style="max-width: 100%;">
  </a>
</p>

<p align="center">
  PIL × DAT - Pillow extension for AI-based image upscaling.
</p>

<p align="center">
    <a href="https://github.com/iLoveDataJjia/pillow-dat/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ilovedatajjia/pillow-dat" alt="License"></a>
</p>

---

## Installation

For PyPI:

```bash
pip install pillow-dat
```

For Conda:

```bash
conda install -c conda-forge pillow-dat
```

## Get started

- 1. Download `DAT_light_x2.pth` model weight from [Google Drive](https://drive.google.com/drive/folders/1ro8bAZxrIEm03eE-7Lc15q9cwE3CJ-oh?usp=sharing). We highly recommend utilizing `DAT_light_x*.pth` models due to their lightweight design and exceptional speed.
- 2. 🎉 Then you are all set to upscale your images:

```python
from PIL.Image import open
from PIL_DAT.Image import upscale

image = open("./lumine.png")
image = upscale(lumine_image, "./DAT_light_x2.pth", 2)
image.show()
```

## Contribution

Please install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

Please install VSCode extensions:

- Black Formatter
- isort
- Python
- Pylance

To install dependencies on a given `pillow-dat` Python environnement:

```bash
conda env create --file environment.yml
```

To update this environment:

```bash
conda env update --file environment.yml --prune
```

To run unit tests:

```bash
python -m pytest
```

## Acknowledgement

This library is founded upon the pioneering research paper, ["Dual Aggregation Transformer for Image Super-Resolution"](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Dual_Aggregation_Transformer_for_Image_Super-Resolution_ICCV_2023_paper.pdf).

```
@inproceedings{chen2023dual,
    title={Dual Aggregation Transformer for Image Super-Resolution},
    author={Chen, Zheng and Zhang, Yulun and Gu, Jinjin and Kong, Linghe and Yang, Xiaokang and Yu, Fisher},
    booktitle={ICCV},
    year={2023}
}
```
