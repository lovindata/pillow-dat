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

## Get started

```python
from PIL.Image import open

from PIL_DAT.Image import upscale

lumine_image = open(".github/lumine.png")
lumine_image = upscale(lumine_image, 2)
lumine_image.show()
```

_Remark_: We strongly advocate for the utilization of `DAT light` models owing to their streamlined design and outstanding speed performance. However, should you opt for alternative models, please note that `*.pth` model weights can be accessed via [Google Drive](https://drive.google.com/drive/folders/1ro8bAZxrIEm03eE-7Lc15q9cwE3CJ-oh?usp=sharing).

## Example

|                                            Input (lumine.png)                                            |                                               Output                                                |                                                Bicubic                                                |
| :------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| ![Input (lumine.png)](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine.png?raw=true) | ![Output](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine_output.png?raw=true) | ![Bicubic](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine_bicubic.png?raw=true) |

## Contribution

Please install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

Please install VSCode extensions:

- Black Formatter
- isort
- Python
- Pylance

To create or update the `pillow-dat` Python environment:

```bash
conda env create --file environment.yml
```

```bash
conda env update --file environment.yml --prune
```

To install dependencies:

```bash
poetry install
```

To run unit tests:

```bash
pytest
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
