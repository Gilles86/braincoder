# Writing a custom encoding model

The minimum is a subclass with `parameter_labels`, one `transformations` entry per
label, and `_basis_predictions`. This template is exercised by
`scripts/check_skill_claims.py`.

```python
import numpy as np
from keras import ops
from braincoder.models import EncodingModel
from braincoder.stimuli import Stimulus

class SigmoidModel(EncodingModel):
    parameter_labels = ['x0', 'slope', 'amplitude', 'baseline']
    stimulus_type = Stimulus

    def __init__(self, **kwargs):
        if not hasattr(self, 'transformations'):
            self.transformations = ['identity', 'identity', 'aggressive_softplus', 'identity']
        super().__init__(**kwargs)

    def _basis_predictions(self, paradigm, parameters):
        x = paradigm[..., None, 0]                      # (batch, time, 1)
        x0, slope, amplitude, baseline = [parameters[:, None, :, i] for i in range(4)]   # (batch, 1, voxels)
        z = ops.clip(-slope * (x - x0), -30., 30.)
        return baseline + amplitude / (1. + ops.exp(z))  # (batch, time, voxels)
```

## The contract

- A model holds configuration only: no paradigm, data, parameters, weights or noise model on `self`.
- `paradigm` has shape `(batch, time, stimulus_dims)`; `parameters` has shape
  `(batch, voxels, n_pars)` in **native** (constrained) space.
- Return `(batch, time, voxels)`. With weights, voxels are basis functions and the base
  class applies the weights.
- Use `keras.ops` only. No `tf.*`, no `float()`, no `convert_to_numpy` in this path, or
  JAX tracing breaks and TF loses gradients.
- Guard exponents, divisions and logs. A NaN in one voxel's gradient stays in that voxel
  under per-element Adam, but poisons any shared parameter.

## `transformations`: the classic bug

`_transform_parameters_forward` loops over `self.transformations`, one column per entry.
**If you add a label but do not add a transformation, columns are dropped.** In the
best case `fit()` crashes with a shape error (`Incompatible shapes: [60,4] vs. [60,5]`);
`predict()` still works because it never transforms, so the bug only shows at fit time.
This silently corrupted real models: `DifferenceOfGaussiansPRF2D` lost `srf_amplitude`
and `srf_size` until 0.6.1, and a project's custom `_transform_parameters_forward2`
stopped being wired up after a refactor.

Allowed entries:

- `'identity'`, `'softplus'`, `'aggressive_softplus'`, `'sigmoid'`
- a `(forward, backward)` tuple of callables, for bounded parameters

A subclass of a model that already set `transformations` must extend them, and the
`hasattr` guard means the parent's list wins if it is set first:

```python
model.transformations = list(model.transformations) + ['softplus']
```

Check with a round trip: forward(backward(p)) must return p, with all columns.
`tests/test_hrf_parameter_transforms.py` does this for the shipped models.

## Adding an HRF

The mixin comes **first** in the bases, and both `__init__`s are called explicitly:

```python
class SigmoidModelWithHRF(HRFEncodingModel, SigmoidModel):
    def __init__(self, hrf_model=None, flexible_hrf_parameters=False, **kwargs):
        SigmoidModel.__init__(self, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)
```

- The mixin sets amplitude to 1 and baseline to 0 before convolving, then reapplies them,
  **but only for columns labeled exactly `amplitude` and `baseline`.** Other names
  (`rf_amplitude`, `bold_baseline`) need their own handling, as in the DN model.
- `flexible_hrf_parameters=True` appends the HRF labels at the end. Write
  `_transform_parameters_forward/backward` to split off the last
  `len(hrf_model.parameter_labels)` columns and send them through
  `hrf_model._transform_parameters_*`, as `GaussianPRF2DWithHRF` does. A wrong split here
  was the DN bug fixed in 0.6.1: the delay followed the BOLD baseline, and nothing errored.
- Only `SPMHRFModel` and `SPMHRFDerivativeModel` exist in `braincoder.hrf`.

## Before you trust it

1. **Round trip** the transformations (above).
2. **Parameter recovery:** simulate from known parameters with realistic noise, fit,
   correlate. Use a realistic SNR: see the `rf_normalization` note in `SKILL.md`.
3. **Default-preserving:** new knobs get a default that reproduces old fits exactly, and
   live on the base class, not copied per subclass.
4. **Check that new arguments are wired in**: passing an extreme value should change the
   output.
5. If grid fitting: `fit_grid` needs a range for every label and raises `NotImplementedError`
   on `fixed_pars`; `refine_baseline_and_amplitude` needs columns named `baseline` and
   `amplitude`. For other names, go straight to `fit(init_pars=…)`.
6. `DataFrame.rename` returns a new frame. Moving from Gaussian to DN needs
   `init = init.rename(columns={'baseline': 'bold_baseline', 'amplitude': 'rf_amplitude'})`,
   and post-fit R² must use the DN model, not the Gaussian one. Both mistakes crashed
   only after a 30-minute GPU fit, so test a warm start on 100 voxels first.
