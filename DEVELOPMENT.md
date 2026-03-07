# Development Notes

## Where we are

Branch: `dev/refactor` (not yet merged to `main`)

---

## What has been done

### 1. Refactoring
`braincoder/models.py` (2200 lines) and `braincoder/optimize.py` (1500 lines) were split into proper subpackages:

```
braincoder/models/
    base.py       — EncodingModel, EncodingRegressionModel, HRFEncodingModel
    prf_1d.py     — GaussianPRF, VonMisesPRF, LogGaussianPRF, AlphaGaussianPRF, ...
    prf_2d.py     — GaussianPRF2D, DifferenceOfGaussiansPRF2D, ...
    linear.py     — LinearModel, LinearModelWithBaseline, DiscreteModel, ...

braincoder/optimize/
    parameter_fitter.py
    weight_fitter.py
    residual_fitter.py
    stimulus_fitter.py
```

All old import paths still work (`from braincoder.models import GaussianPRF` etc.).

### 2. Bug fixes
- `VonMisesPRF` — crashed on construction due to unexpected `n_dimensions` kwarg
- `GaussianPRFWithHRF` / `LogGaussianPRFWithHRF` — wrong MRO (method resolution order) broke the HRF mixin

### 3. Tests
101 tests across 5 files, all passing:

| File | What it covers |
|------|---------------|
| `tests/test_models.py` | GaussianPRF, GaussianPRF2D, LinearModel, ParameterFitter, WeightFitter |
| `tests/test_models_extended.py` | VonMisesPRF, LogGaussianPRF, AlphaGaussianPRF, GaussianPRFWithHRF, GaussianPRF2DAngle, DifferenceOfGaussiansPRF2D, ResidualFitter, simulate() edge cases |
| `tests/test_hrf.py` | SPMHRFModel, spm_hrf function, HRF convolution |
| `tests/test_utils.py` | format_paradigm/parameters/data/weights, gamma_pdf |

Run them locally:
```bash
conda activate braincoder
pytest -v
```

### 4. CI on GitHub
Every push to `main` or `dev/**` (and every PR to `main`) runs the test suite automatically on Python 3.9 and 3.10.

Status of last run: https://github.com/Gilles86/braincoder/actions

### 5. Fixed package dependencies
`pyproject.toml` now correctly declares all required packages:
`tensorflow`, `tensorflow-probability`, `tf_keras`, `scipy`, `tqdm`, `patsy`, `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## What is planned next

### Priority 1 — Merge `dev/refactor` to `main`
The branch is stable and CI is green. Time to ship it.

### Priority 2 — Validation cleanup
Right now, passing wrong inputs to a model (wrong parameter column names, wrong paradigm shape, etc.) produces cryptic TensorFlow errors deep in graph execution. The plan is to add clear, early `ValueError` messages.

**The plan is saved in full detail at: `/Users/gdehol/.claude/plans/starry-bouncing-sunset.md`**

Short summary of what changes:

| File | Change |
|------|--------|
| `braincoder/utils/formatting.py` | Add `validate_parameter_columns()` helper |
| `braincoder/utils/__init__.py` | Export it |
| `braincoder/models/base.py` | Use validator in `__init__` and `_get_parameters`; add clear error when `parameters=None` at predict time |
| `braincoder/models/linear.py` | Delete 6 lines of duplicated paradigm setup |
| `braincoder/models/prf_2d.py` | Add dimension check in `GaussianPRF2D` before it crashes on `paradigm.shape[1]` |
| `tests/test_validation.py` | 8 new tests |

To continue: open this repo and tell Claude "continue the validation cleanup plan".

### Priority 3 — Known pre-existing bugs (not yet fixed)
- `SPMHRFDerivativeModel` — crashes on construction (`bounded_sigmoid_transform` undefined)
- `CustomHRFModel` — stub class, no `get_hrf()` method
- 5 legacy test files (`tests/test_glm.py` etc.) — import classes that were removed long ago; currently ignored by pytest config

---

## How to run things

```bash
# Run all tests
conda activate braincoder
pytest -v

# Run just one file
pytest tests/test_models.py -v

# Install in development mode (needed after pulling fresh)
pip install -e ".[test]"

# Check CI status
gh run list --repo Gilles86/braincoder
```
