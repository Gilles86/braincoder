"""Calibration metrics for grid posteriors (e.g. from ``get_stimulus_pdf``).

All functions take a posterior DataFrame — rows = trials, columns = grid
stimuli — and the true stimulus of every trial. Rows are renormalised to sum
to one (discrete masses over the grid), so both the trapezoid-normalised 1-D
densities and the max-normalised multi-dimensional output of
``get_stimulus_pdf`` can be passed directly.

A calibrated posterior gives

* randomised PIT values ~ Uniform(0, 1) (1-D only; SD ≈ 0.289, slightly less
  on a coarse discrete grid), U-shaped = overconfident, ∩-shaped =
  underconfident;
* randomised HPD levels of the truth ~ Uniform(0, 1), i.e. the q-HDI
  contains the truth on a fraction q of trials (any dimensionality);
* a log score that is not below that of a flat posterior.

Randomisation (a uniform draw within the probability mass of the true grid
cell) makes the statistics exactly uniform for a calibrated discrete
posterior instead of biased by ties and grid coarseness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ['posterior_grid', 'marginalize', 'true_stimulus_index', 'log_score',
           'pit', 'hpd_level', 'hdi_coverage', 'calibration_summary',
           'plot_calibration']


def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def posterior_grid(pdf, grid=None):
    """(P, grid): row-normalised masses (n_trials, n_stimuli) and the grid.

    ``grid`` is taken from the columns (numeric Index or MultiIndex) unless
    passed explicitly (needed when ``get_stimulus_pdf`` was called on a
    multi-dimensional grid without ``include_multidimensional_stimulus_index``).
    """
    P = np.asarray(pdf, dtype=np.float64)
    P = np.clip(np.nan_to_num(P), 0, None)
    P = P / P.sum(1, keepdims=True)
    if grid is None:
        cols = pdf.columns
        if isinstance(cols, pd.MultiIndex):
            grid = np.array([np.asarray(c, dtype=float) for c in cols.to_flat_index()])
        elif isinstance(cols, pd.RangeIndex) or cols.dtype == object:
            raise ValueError('Cannot infer the stimulus grid from the columns; '
                             'pass grid=')
        else:
            grid = np.asarray(cols, dtype=float)[:, None]
    grid = np.asarray(grid, dtype=float)
    if grid.ndim == 1:
        grid = grid[:, None]
    if len(grid) != P.shape[1]:
        raise ValueError('grid length does not match the number of columns')
    return P, grid


def marginalize(pdf, dimension, grid=None):
    """Marginal posterior over one stimulus dimension (int or level name)."""
    P, grid = posterior_grid(pdf, grid)
    if not isinstance(dimension, (int, np.integer)):
        dimension = list(pdf.columns.names).index(dimension)
    vals, inv = np.unique(grid[:, dimension], return_inverse=True)
    M = np.zeros((len(P), len(vals)))
    np.add.at(M.T, inv, P.T)
    name = None
    if isinstance(pdf.columns, pd.MultiIndex):
        name = pdf.columns.names[dimension]
    return pd.DataFrame(M, index=pdf.index, columns=pd.Index(vals, name=name))


def true_stimulus_index(true_stimulus, grid):
    """Index of the nearest grid point for every true stimulus."""
    t = np.asarray(true_stimulus, dtype=float)
    if t.ndim == 1:
        if grid.shape[1] != 1:
            raise ValueError('true_stimulus must be (n_trials, n_dims) for a '
                             'multi-dimensional grid')
        t = t[:, None]
    d2 = ((t[:, None, :] - grid[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1)


def log_score(pdf, true_stimulus, grid=None, relative_to_uniform=True):
    """Per-trial log posterior mass of the true grid cell.

    With ``relative_to_uniform`` (default) ``log(1/n_stimuli)`` is subtracted,
    so 0 = as good as a flat posterior and > 0 = informative and calibrated
    enough to beat it.
    """
    P, grid = posterior_grid(pdf, grid)
    ti = true_stimulus_index(true_stimulus, grid)
    ls = np.log(np.clip(P[np.arange(len(P)), ti], 1e-300, None))
    if relative_to_uniform:
        ls = ls + np.log(P.shape[1])
    return ls


def pit(pdf, true_stimulus, grid=None, rng=0):
    """Randomised probability integral transform (1-D posteriors).

    F(s* − 1) + U · p(s*), U ~ U(0, 1). For a multi-dimensional posterior,
    first :func:`marginalize` it.
    """
    P, grid = posterior_grid(pdf, grid)
    if grid.shape[1] != 1:
        raise ValueError('PIT needs a 1-D posterior; marginalize() first')
    order = np.argsort(grid[:, 0])
    P, grid = P[:, order], grid[order]
    ti = true_stimulus_index(true_stimulus, grid)
    cdf = np.cumsum(P, 1)
    below = np.where(ti > 0, cdf[np.arange(len(P)), np.maximum(ti - 1, 0)], 0.)
    u = _rng(rng).uniform(size=len(P))
    return below + u * P[np.arange(len(P)), ti]


def hpd_level(pdf, true_stimulus, grid=None, rng=0):
    """Randomised level of the smallest HDI that contains the truth.

    h = P(p(s) > p(s*)) + U · P(p(s) = p(s*)). The q-HDI covers the truth
    iff h < q. Works for any dimensionality.
    """
    P, grid = posterior_grid(pdf, grid)
    ti = true_stimulus_index(true_stimulus, grid)
    pt = P[np.arange(len(P)), ti][:, None]
    above = (P * (P > pt)).sum(1)
    tie = (P * (P == pt)).sum(1)
    u = _rng(rng).uniform(size=len(P))
    return above + u * tie


def hdi_coverage(pdf, true_stimulus, levels=(0.5, 0.95), grid=None, rng=0):
    """Fraction of trials whose q-HDI contains the truth, for each q."""
    h = hpd_level(pdf, true_stimulus, grid=grid, rng=rng)
    return pd.Series({q: float(np.mean(h < q)) for q in levels}, name='coverage')


def calibration_summary(pdf, true_stimulus, grid=None, levels=(0.5, 0.95), rng=0):
    """One-row summary: log score vs uniform, HDI coverage, PIT SD (1-D),
    posterior mean r / MAE / posterior SD (1-D)."""
    P, g = posterior_grid(pdf, grid)
    out = {'n_trials': len(P),
           'log_score': float(np.mean(log_score(pdf, true_stimulus, grid=grid)))}
    for q, c in hdi_coverage(pdf, true_stimulus, levels=levels, grid=grid, rng=rng).items():
        out[f'coverage_{q:g}'] = c
    if g.shape[1] == 1:
        p = pit(pdf, true_stimulus, grid=grid, rng=rng)
        out['pit_sd'] = float(np.std(p))
        out['pit_tails_5pct'] = float(np.mean((p < .025) | (p > .975)))
        t = np.asarray(true_stimulus, dtype=float).ravel()
        mean = P @ g[:, 0]
        out['posterior_sd'] = float(np.mean(np.sqrt(np.clip(P @ g[:, 0] ** 2 - mean ** 2, 0, None))))
        out['r_posterior_mean'] = float(np.corrcoef(mean, t)[0, 1]) if np.std(mean) > 0 else np.nan
        out['mae_posterior_mean'] = float(np.mean(np.abs(mean - t)))
    return pd.Series(out)


def plot_calibration(pdf, true_stimulus, grid=None, rng=0, axes=None, label=None,
                     levels=np.linspace(0.05, 0.95, 19)):
    """Two-panel diagnostic: coverage vs nominal HDI level, and PIT histogram
    (1-D) or HPD-level histogram (multi-D). Returns the axes."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(7, 3))
    P, g = posterior_grid(pdf, grid)
    h = hpd_level(pdf, true_stimulus, grid=grid, rng=rng)
    cov = [np.mean(h < q) for q in levels]
    axes[0].plot([0, 1], [0, 1], color='0.6', lw=1, ls='--')
    axes[0].plot(levels, cov, marker='o', ms=3, label=label)
    axes[0].set(xlabel='Nominal HDI level', ylabel='Empirical coverage',
                xlim=(0, 1), ylim=(0, 1))
    if g.shape[1] == 1:
        vals, xlabel = pit(pdf, true_stimulus, grid=grid, rng=rng), 'Randomised PIT'
    else:
        vals, xlabel = h, 'Randomised HPD level of truth'
    axes[1].hist(vals, bins=np.linspace(0, 1, 11), density=True, alpha=.6, label=label)
    axes[1].axhline(1, color='0.6', lw=1, ls='--')
    axes[1].set(xlabel=xlabel, ylabel='Density', xlim=(0, 1))
    return axes
