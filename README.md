# Braincoder

**Braincoder** is a package to fit encoding models to neural data (for now fMRI) and
to then *invert* those models to decode stimulus information from neural data. It
wraps stimulus handling, model definition, HRF convolution, and optimization into a
single [Keras 3](https://keras.io)-based toolkit that runs on TensorFlow, JAX, or PyTorch.

## Highlights

- **Composable models.** Pick from ready-made encoding models (Gaussian PRF, HRF-aware, linear) or subclass `EncodingModel` to implement your own.
- **End-to-end workflow.** Utilities for simulation, fitting, and decoding stay in pandas/tensor-friendly formats.
- **Backend-agnostic.** Runs on TensorFlow, JAX, or PyTorch via Keras 3.

## Links

- Source code: https://github.com/Gilles86/braincoder
- Documentation: https://braincoder-devs.github.io/

## Installation

The quickest way to get started is with conda/mamba. Clone the repo, then create the environment:

```bash
git clone https://github.com/Gilles86/braincoder.git
cd braincoder
conda env create -f environment.yml
conda activate braincoder
```

This installs braincoder in editable mode (`pip install -e .`) along with a TensorFlow backend and dev tools. See the [installation docs](https://braincoder-devs.github.io/installation.html) for other backends (JAX, PyTorch) and pip-only setups.

## Quick start

```python
import numpy as np
import pandas as pd
from braincoder.models import GaussianPRF

paradigm = pd.DataFrame({"x": np.linspace(-5, 5, 100)},
                        index=pd.Index(np.arange(100), name="time"))
parameters = pd.DataFrame({"mu": [0.0, 1.0], "sd": [1.0, 2.0], "amplitude": [1.0, 1.0], "baseline": [0.0, 0.0]},
                          index=pd.Index(["v1", "v2"], name="voxel"))

model = GaussianPRF()
predicted = model.predict(paradigm=paradigm, parameters=parameters)
```

See the [tutorials](https://braincoder-devs.github.io/tutorial/index.html) for full fitting and decoding pipelines.

## How to cite

de Hollander, G., Renkert, M., Ruff, C. C., & Knapen, T. H. (2024). *Braincoder: A package for fitting encoding models to neural data and decoding stimulus features*. Zenodo. DOI: [10.5281/zenodo.10778413](https://doi.org/10.5281/zenodo.10778413).

```bibtex
@software{deHollander2024braincoder,
  author    = {Gilles de Hollander and Maike Renkert and Christian C. Ruff and Tomas H. Knapen},
  title     = {braincoder: A package for fitting encoding models to neural data and decoding stimulus features},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.10778413},
  url       = {https://github.com/Gilles86/braincoder}
}
```
