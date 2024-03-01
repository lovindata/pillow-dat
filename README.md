# DAT-light packaged

## Installation

Please install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

Please install VSCode extensions:

- Black Formatter
- isort
- Python
- Pylance

To install dependencies on a given `dat-light-packaged` Python environnement:

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

## References

```
@inproceedings{chen2023dual,
    title={Dual Aggregation Transformer for Image Super-Resolution},
    author={Chen, Zheng and Zhang, Yulun and Gu, Jinjin and Kong, Linghe and Yang, Xiaokang and Yu, Fisher},
    booktitle={ICCV},
    year={2023}
}
```
