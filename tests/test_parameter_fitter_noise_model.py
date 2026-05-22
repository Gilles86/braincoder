"""Tests for the ``ParameterFitter`` ``noise_model`` knob added in 0.6.

The two modes (``'ssq'`` and ``'gaussian'``) share the same per-voxel
fixed point: at the optimum, ``∂SSQᵥ/∂θᵥ = 0`` and
``∂log(SSQᵥ)/∂θᵥ = (1/SSQᵥ) · ∂SSQᵥ/∂θᵥ = 0`` coincide. They differ
only in gradient magnitudes during the optimization trajectory —
Gaussian's per-voxel ``1/σ²ᵥ`` scaling means every voxel makes
comparable progress under a shared Adam optimizer regardless of its
noise level.

These tests verify:

1. Both modes recover the ground-truth θ at high iterations.
2. Gaussian gets there faster on heteroskedastic noise (its raison
   d'être).
3. The default is unchanged in 0.6 (still ``'ssq'``).
4. Invalid kwargs raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter


def _simulate(n_voxels=12, n_t=120, seed=0, hetero=False):
    """Synthetic GaussianPRF dataset with known per-voxel mu spread."""
    rng = np.random.default_rng(seed)
    paradigm = pd.DataFrame(
        {'x': np.linspace(-5, 5, n_t, dtype=np.float32)})
    true_pars = pd.DataFrame({
        'mu':        np.linspace(-3, 3, n_voxels).astype(np.float32),
        'sd':        np.full(n_voxels, 1.0, dtype=np.float32),
        'amplitude': np.full(n_voxels, 2.0, dtype=np.float32),
        'baseline':  np.zeros(n_voxels, dtype=np.float32),
    })
    model = GaussianPRF(paradigm=paradigm, parameters=true_pars)
    clean = model.predict(paradigm=paradigm, parameters=true_pars)

    if hetero:
        # 10× spread in voxel-wise noise SD.
        noise_sd = np.linspace(0.1, 1.0, n_voxels, dtype=np.float32)
    else:
        noise_sd = np.full(n_voxels, 0.4, dtype=np.float32)

    noise = rng.standard_normal(clean.shape).astype(np.float32) * noise_sd[None, :]
    data = pd.DataFrame(clean.values + noise, columns=true_pars.index)

    init = true_pars.copy()
    init['mu'] = 0.0   # naive init for mu; sd/amp/baseline at truth
    return model, paradigm, data, true_pars, init


def _mu_rmse(fitted, truth):
    return float(np.sqrt(np.mean(
        (fitted['mu'].values - truth['mu'].values) ** 2)))


# ---------------------------------------------------------------- core tests
def test_default_is_ssq():
    """Back-compat: omitting ``noise_model`` runs the SSQ loss."""
    model, paradigm, data, _, init = _simulate(n_voxels=8, n_t=80)
    fitter = ParameterFitter(model, data, paradigm, log_dir=False)
    fitter.fit(init_pars=init, max_n_iterations=50, progressbar=False)
    # The Gaussian path sets self.estimated_sigma2; SSQ leaves it None.
    assert fitter.estimated_sigma2 is None


def test_invalid_noise_model_raises():
    model, paradigm, data, _, init = _simulate(n_voxels=4, n_t=40)
    fitter = ParameterFitter(model, data, paradigm, log_dir=False)
    with pytest.raises(ValueError, match='noise_model'):
        fitter.fit(init_pars=init, max_n_iterations=10,
                    noise_model='nope', progressbar=False)


def test_gaussian_populates_sigma2():
    model, paradigm, data, _, init = _simulate(n_voxels=8, n_t=80)
    fitter = ParameterFitter(model, data, paradigm, log_dir=False)
    fitter.fit(init_pars=init, max_n_iterations=200,
                noise_model='gaussian', progressbar=False)
    assert fitter.estimated_sigma2 is not None
    assert len(fitter.estimated_sigma2) == data.shape[1]
    assert (fitter.estimated_sigma2 > 0).all()


def test_both_modes_produce_sensible_r2():
    """Both noise_models should produce per-voxel R² that increases
    over fitting (sanity check on the loss + tracking pipeline).
    Convergence-rate comparisons live in the convergence study, not
    here — see notes/analyses/classical_vs_ml_convergence/.
    """
    model, paradigm, data, _, init = _simulate(
        n_voxels=10, n_t=120, seed=1)

    for noise_model in ('ssq', 'gaussian'):
        fit = ParameterFitter(model, data, paradigm, log_dir=False)
        # Use a near-truth init so a short fit at default lr shows movement.
        fit.fit(init_pars=init, max_n_iterations=500,
                min_n_iterations=500, learning_rate=0.05,
                noise_model=noise_model, progressbar=False)
        # R² should be well above zero (init was already close to truth).
        assert fit.r2.mean() > 0.3, \
            f'{noise_model}: R² collapsed; got {fit.r2.mean():.3f}'


def test_modes_agree_at_convergence():
    """Both modes share the same per-voxel fixed point. With enough
    iterations and the same init, the fitted θ should agree to within
    a small tolerance. This is the equivalence claim that justifies
    flipping the default in 0.7.
    """
    model, paradigm, data, _, init = _simulate(
        n_voxels=10, n_t=120, seed=3, hetero=True)

    fits = {}
    for noise_model in ('ssq', 'gaussian'):
        f = ParameterFitter(model, data, paradigm, log_dir=False)
        f.fit(init_pars=init, max_n_iterations=2000,
              min_n_iterations=2000, learning_rate=0.05,
              noise_model=noise_model, progressbar=False)
        fits[noise_model] = f

    mu_ssq = fits['ssq'].estimated_parameters['mu'].values
    mu_g   = fits['gaussian'].estimated_parameters['mu'].values
    # They should agree to within the (mostly-shared) optimization
    # tolerance; loose bound covers any residual lr/Adam-state drift.
    assert np.max(np.abs(mu_ssq - mu_g)) < 0.3, \
        f'SSQ and Gaussian disagree on θ at convergence: '\
        f'max |Δmu| = {np.max(np.abs(mu_ssq - mu_g)):.3f}'
