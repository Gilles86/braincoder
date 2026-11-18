"""Divisive normalization models with a surround baseline of exactly 0 (GitHub PR #7)."""
import numpy as np
import pandas as pd
import pytest

from braincoder.hrf import SPMHRFModel
from braincoder.models import DivisiveNormalizationGaussianPRF2D, DivisiveNormalizationGaussianPRF2DWithHRF
from braincoder.utils import get_rsq

x = np.linspace(-5, 5, 11)
GRID = np.stack([a.ravel() for a in np.meshgrid(x, x)], 1).astype(np.float32)
IMAGES = np.zeros((40, len(GRID)), np.float32)
IMAGES[5:20, 40:60] = 1.


def _pars(model, surround_baseline):
    values = dict(x=0., y=0., sd=1., rf_amplitude=1., srf_amplitude=.5, srf_size=2., neural_baseline=1.,
                  surround_baseline=surround_baseline, bold_baseline=0.)
    return pd.DataFrame([{k: values[k] for k in model.parameter_labels}]).astype(np.float32)


@pytest.mark.parametrize('with_hrf', [False, True])
def test_zero_surround_baseline_gives_finite_predictions(with_hrf):
    model = (DivisiveNormalizationGaussianPRF2DWithHRF(GRID, hrf_model=SPMHRFModel(tr=1.)) if with_hrf
             else DivisiveNormalizationGaussianPRF2D(GRID))
    pred = model.predict(IMAGES, _pars(model, 0.))
    assert np.isfinite(pred.values).all()


def test_zero_surround_baseline_does_not_give_a_perfect_r2():
    model = DivisiveNormalizationGaussianPRF2DWithHRF(GRID, hrf_model=SPMHRFModel(tr=1.))
    pred = model.predict(IMAGES, _pars(model, 0.))
    data = pd.DataFrame(np.random.default_rng(0).normal(size=pred.shape), index=pred.index, columns=pred.columns)
    assert float(get_rsq(data, pred).iloc[0]) < 1.


def test_positive_surround_baseline_is_unchanged():
    model = DivisiveNormalizationGaussianPRF2D(GRID)
    blank = model.predict(np.zeros((1, len(GRID)), np.float32), _pars(model, 2.)).values
    np.testing.assert_allclose(blank, 1. / 2.)
