from helpers import Bound, grid_for
import numpy as np
import pandas as pd
import pytest

from braincoder.hrf import SPMHRFModel
from braincoder.models import GaussianPRF2D, DivisiveNormalizationGaussianPRF2DWithHRF

G = np.linspace(-8, 8, 81)
GRID = np.stack(np.meshgrid(G, G), -1).reshape(-1, 2).astype(np.float32)
PIX = (G[1] - G[0]) ** 2


def rf(norm, sd=1.3):
    pars = pd.DataFrame(dict(x=[0.], y=[0.], sd=[sd], baseline=[0.], amplitude=[1.])).astype(np.float32)
    m = Bound(GaussianPRF2D(grid_coordinates=GRID, rf_normalization=norm), np.zeros((2, len(GRID)), np.float32), pars)
    return m.get_rf().ravel()


def test_legacy_matches_previous_formula():
    sd = 1.3
    peak1 = np.exp(-(GRID ** 2).sum(1) / (2 * sd ** 2))
    np.testing.assert_allclose(rf('legacy', sd), peak1 / (sd * np.sqrt(2 * np.pi) / PIX), rtol=1e-5, atol=1e-8)


def test_peak_is_one():
    np.testing.assert_allclose(rf('peak').max(), 1.0, rtol=1e-6)


def test_area_integrates_to_one():
    np.testing.assert_allclose(rf('area').sum(), 1.0, rtol=1e-3)


def test_unknown_normalization_raises():
    with pytest.raises(ValueError):
        Bound(GaussianPRF2D(grid_coordinates=GRID, rf_normalization='max'), np.zeros((2, len(GRID)), np.float32), None)


def test_dn_peak_reparameterises_legacy():
    rng = np.random.default_rng(0)
    paradigm = (rng.random((60, len(GRID))) < .1).astype(np.float32)
    hrf = SPMHRFModel(tr=1.6)
    legacy = Bound(DivisiveNormalizationGaussianPRF2DWithHRF(grid_coordinates=GRID, hrf_model=hrf), paradigm, None)
    peak = Bound(DivisiveNormalizationGaussianPRF2DWithHRF(grid_coordinates=GRID, hrf_model=hrf, rf_normalization='peak'), paradigm, None)
    p = pd.DataFrame(dict(x=[1.], y=[-.5], sd=[1.2], rf_amplitude=[2.], srf_amplitude=[.3], srf_size=[2.5],
                          neural_baseline=[.5], surround_baseline=[1.5], bold_baseline=[.1])).astype(np.float32)
    w = lambda s: PIX / (s * np.sqrt(2 * np.pi))
    q = p.copy()
    q['rf_amplitude'] = p.rf_amplitude * w(p.sd)
    # surround weight is srf_amplitude * |rf_amplitude| * w(sd * srf_size)
    q['srf_amplitude'] = p.srf_amplitude * abs(p.rf_amplitude) * w(p.sd * p.srf_size) / q.rf_amplitude.abs()
    np.testing.assert_allclose(np.asarray(peak.predict(paradigm=paradigm, parameters=q.astype(np.float32))),
                               np.asarray(legacy.predict(paradigm=paradigm, parameters=p)), rtol=1e-4, atol=1e-6)
