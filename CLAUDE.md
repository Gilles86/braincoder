# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Braincoder is a TensorFlow-based neuroimaging package for fitting encoding models to fMRI data and decoding stimuli from neural responses. It models how neurons respond to stimuli via Population Receptive Fields (PRFs) and supports HRF (hemodynamic response function) convolution for fMRI.

## Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_gauss_rf.py

# Run a specific test
pytest tests/test_glm.py::test_function_name
```

No linting configuration is present in the repo.

## Architecture

### Core Module Layout

- **`braincoder/models.py`** — All encoding models (~2200 lines). The central module.
- **`braincoder/optimize.py`** — All optimizer/fitter classes (~1500 lines).
- **`braincoder/hrf.py`** — HRF kernels and convolution utilities.
- **`braincoder/stimuli.py`** — Stimulus representation base classes.
- **`braincoder/barstimuli.py`** — Bar stimulus fitting (specialized stimulus type).
- **`braincoder/utils/`** — Data formatting (`formatting.py`), math (`math.py`), stats (`stats.py`), MCMC (`mcmc.py`), visualization (`visualize.py`).

### Model Hierarchy

`EncodingModel` is the abstract base. Subclasses implement `_basis_predictions(paradigm, parameters)` which returns predicted neural responses for a given stimulus paradigm.

Key model families:
- `GaussianPRF` → `LogGaussianPRF`, `VonMisesPRF`, `AlphaGaussianPRF`, `GaussianPRF2D`, `DifferenceOfGaussiansPRF`
- `EncodingRegressionModel` → `RegressionGaussianPRF`, `RegressionAlphaGaussianPRF`
- **HRF Mixin**: `HRFEncodingModel` is a mixin added via multiple inheritance (e.g., `class GaussianPRFWithHRF(GaussianPRF, HRFEncodingModel)`)

### Two-Level API Pattern

All models expose two tiers:
- **TensorFlow level**: `_basis_predictions()`, `_predict()` — decorated with `@tf.function`, operate on tensors, called during optimization
- **Pandas level**: `predict()`, `simulate()` — accept/return DataFrames with named voxels, parameters, and time indices

Data conversion utilities in `utils/formatting.py` bridge these layers (`format_data()`, `format_paradigm()`, `format_parameters()`, `format_weights()`).

### Optimization Pipeline

Fitting follows a sequential chain in `optimize.py`:

1. **`ParameterFitter`** — Iterative Adam optimizer for PRF parameters (x, y, sd, etc.)
2. **`WeightFitter`** — Closed-form least-squares for voxel weights given fixed parameters
3. **`ResidualFitter`** — Fits noise model (Gaussian or Student-t) to residuals
4. **`StimulusFitter`** / **`BarStimulusFitter`** — Decodes stimuli from fitted model + observed data

### Parameter Constraints

TensorFlow-Probability bijectors (`softplus`, `sigmoid`, periodic transforms) are used to constrain parameters to valid ranges. Each model class defines its own bijectors for its parameter space.

### Data Conventions

- **Paradigm**: stimulus timeseries, indexed by time (DataFrame)
- **Data**: neural timeseries, columns = voxels, rows = timepoints (DataFrame)
- **Parameters**: voxel-wise PRF parameters, index = voxels (DataFrame)
- **Weights**: basis function weights per voxel (DataFrame)

Tests in `tests/` are integration-style: simulate data from known parameters → fit model → assert recovered parameters correlate with ground truth.
