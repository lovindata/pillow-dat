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

|                                            Input (lumine.png)                                            |                                               DAT light (x2)                                                |                                                Bicubic (x2)                                                |
| :------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: |
| ![Input (lumine.png)](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine.png?raw=true) | ![DAT light (x2)](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine_output.png?raw=true) | ![Bicubic (x2)](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine_bicubic.png?raw=true) |

## Benchmarks

### Speed

Performance benchmarks have been conducted on a computing system equipped with an `Intel(R) CORE(TM) i7-9750H CPU @ 2.60GHz processor`, accompanied by a `2 × 8 Go at 2667MHz RAM` configuration. Below are the recorded results:

|  _In seconds_  | 320 × 320 | 640 × 640 | 960 × 960 | 1280 × 1280 |
| :------------: | :-------: | :-------: | :-------: | :---------: |
| DAT light (x2) |   13.7    |   54.9    |   127.2   |    299.3    |
| DAT light (x3) |   13.2    |   56.5    |     -     |      -      |
| DAT light (x4) |   12.8    |   56.6    |     -     |      -      |

The results were compared against the renowned `OpenCV` library, utilizing its `EDSR` model known for delivering superior image quality.

| _In seconds_ | 320 × 320 | 640 × 640 | 960 × 960 | 1280 × 1280 |
| :----------: | :-------: | :-------: | :-------: | :---------: |
|  EDSR (x2)   |   25.6    |   112.9   |   264.1   |    472.8    |
|  EDSR (x3)   |   24.3    |   112.5   |     -     |      -      |
|  EDSR (x4)   |   23.6    |   111.2   |     -     |      -      |

_Remark_: All speed benchmark results presented here are reproducible. For detailed implementation, please refer to the following files: [benchmark_speed_dat_light.py](https://github.com/iLoveDataJjia/pillow-dat/blob/main/benchmarks/benchmark_speed_dat_light.py) and [benchmark_speed_edsr.py](https://github.com/iLoveDataJjia/pillow-dat/blob/main/benchmarks/benchmark_speed_edsr.py).

### Quality

|                                               DAT light (x2)                                                |                                              EDSR (x2)                                               |
| :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
| ![DAT light (x2)](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine_output.png?raw=true) | ![EDSR (x2)](https://github.com/iLoveDataJjia/pillow-dat/blob/main/.github/lumine_edsr.png?raw=true) |

_Remark_: All quality benchmark results presented here are reproducible. For detailed implementation, please refer to the following files: [example.py](https://github.com/iLoveDataJjia/pillow-dat/blob/main/examples/example.py) and [benchmark_quality_edsr.py](https://github.com/iLoveDataJjia/pillow-dat/blob/main/benchmarks/benchmark_quality_edsr.py).

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
