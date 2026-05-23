"""
============================================================
Expected decoding uncertainty via simulate + decode
============================================================

Fisher information gives a *local* lower bound on the variance of any
unbiased estimator of a stimulus from a fitted encoding model. It's
quick to compute and useful as a theoretical bound, but it ignores two
things you usually care about in practice:

* the actual prior / stimulus distribution you'll evaluate on,
* the *bias* of the maximum-a-posteriori or posterior-mean estimator
  (which Fisher info, being a Cramér–Rao bound, says nothing about).

A more direct alternative is to **simulate from the fitted model** and
**re-decode** the simulated responses through the same model. The
distribution of the posterior-mean estimate across many simulated
trials at each stimulus value tells you:

* the **bias** :math:`E[\\hat{s}] - s`, and
* the **expected uncertainty** :math:`\\sqrt{\\mathrm{Var}[\\hat{s}]}`,

both as functions of the stimulus. This is the procedure used in
``neural_priors/encoding_model/get_expected_uncertainty.py``; the
example below ports it to public ``braincoder`` APIs on the bundled
demo dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.utils.data import load_pratcarrabin2025_npc
from braincoder.utils.stats import (
    fit_r2_mixture,
    get_rsq,
)

# %%
# Load the demo + select voxels
# -----------------------------------------------------------------
# Same voxel-selection step as the first two examples — kept self-contained.

bundle = load_pratcarrabin2025_npc()
r2 = bundle['r2']
paradigm_all = bundle['paradigm']
data_all = bundle['data']

r2_wb = bundle['r2_wholebrain'].get_fdata()
mask_wb = bundle['brain_mask'].get_fdata().astype(bool)
fit = fit_r2_mixture(r2_wb[mask_wb])
# Noise μ + 2σ on the logit scale — see example 1 for the rationale.
threshold = 1.0 / (1.0 + np.exp(-(fit['noise_mu'] + 2 * fit['noise_sigma'])))
keep = r2.index[r2 > threshold]
print(f'Kept {len(keep)} voxels (within-sample R² ≥ {threshold:.3f})')

# Restrict to wide-range trials and z-score within the selected voxel set.
is_wide = paradigm_all['range'].values == 'wide'
paradigm = paradigm_all.loc[is_wide].reset_index(drop=True)
data = data_all.loc[is_wide, keep].reset_index(drop=True).astype(np.float32)
data = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, 1)
print(f'Data: {data.shape} (trials × voxels)')

# %%
# Fit the encoding model on *all* trials
# -----------------------------------------------------------------
# For an uncertainty estimate we want the model's best guess at the
# true tuning curves — no held-out trials. The expected-uncertainty
# calculation only needs ``parameters`` and the noise model.

model = LogGaussianPRF(parameterisation='mu_sd_natural')
fitter = ParameterFitter(model, data, paradigm[['n']])

mu_grid   = np.linspace(10, 40, 16, dtype=np.float32)
sd_grid   = np.array([5., 10., 20.], dtype=np.float32)
amp_grid  = np.array([0.5, 1.0, 2.0], dtype=np.float32)
base_grid = np.array([0.0], dtype=np.float32)
init = fitter.fit_grid(mu_grid, sd_grid, amp_grid, base_grid)
pars = fitter.fit(max_n_iterations=600, init_pars=init,
                   noise_model='gaussian', learning_rate=0.05,
                   progressbar=False)

predictions = model.predict(paradigm=paradigm[['n']], parameters=pars)
predictions.index = data.index
train_r2 = get_rsq(data, predictions)
print(f'Train R² — median {np.nanmedian(train_r2):.3f}, '
      f'p90 {np.nanpercentile(train_r2, 90):.3f}')

# %%
# Fit the noise model
# -----------------------------------------------------------------
# Same Student-t noise model as notebook 2 (without geodesic
# regularisation here, to keep the example focused). ``init_pseudoWWT``
# computes the W Wᵀ template over the stimulus grid we'll later use
# for decoding.

stim_grid = pd.DataFrame({'n': np.arange(10, 41, 1, dtype=np.float32)})
stim_grid.index.name = 'stimulus'
model.init_pseudoWWT(stim_grid, pars)

resid_fitter = ResidualFitter(model, data, paradigm[['n']], parameters=pars)
omega, dof = resid_fitter.fit(method='t', init_dof=10.0,
                                max_n_iterations=600, progressbar=False)
print(f'Fitted dof: {dof:.1f}')

# %%
# Simulate, decode, and aggregate in one call
# -----------------------------------------------------------------
# :meth:`EncodingModel.get_expected_uncertainty` wraps the whole
# pipeline — for every stimulus in ``stim_grid`` it (1) simulates
# ``n_simulations`` noise-perturbed response vectors from
# ``(parameters, omega, dof)``, (2) decodes each via
# :meth:`get_stimulus_pdf`, (3) computes the posterior mean, and (4)
# aggregates per true stimulus. It returns a DataFrame with
# ``mean_E, var_E, mean_error, mean_abs_error, n_sims``.

n_repeats = 200
print(f'Simulating {n_repeats} trials per numerosity '
      f'({len(stim_grid)} numerosities) …')
summary = model.get_expected_uncertainty(
    stim_grid['n'].values.astype(np.float32),
    omega=omega, dof=dof, parameters=pars,
    n_simulations=n_repeats, progress=True,
)
# Add convenience columns the plot below reads off directly.
summary['std_decoded'] = np.sqrt(summary['var_E'])
print(summary.round(3).head())

# %%
# Plot bias + expected uncertainty
# -----------------------------------------------------------------
# Per true numerosity: the bias of the posterior-mean estimator
# (``mean_E - true``) and the spread of the posterior mean across
# repeats (``√var_E``).

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(summary.index, summary['mean_E'], 'o-', color='#1f77b4',
         label='Mean decoded')
ax.fill_between(summary.index,
                 summary['mean_E'] - summary['std_decoded'],
                 summary['mean_E'] + summary['std_decoded'],
                 color='#1f77b4', alpha=0.2,
                 label='± std across repeats')
lim = (summary.index.min() - 1, summary.index.max() + 1)
ax.plot(lim, lim, 'k:', lw=1, label='Identity')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('True numerosity')
ax.set_ylabel(r'Decoded $\hat{n}$')
ax.set_title('Bias of posterior mean')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(summary.index, summary['std_decoded'], 'o-', color='#d62728',
         label=r'$\sqrt{\mathrm{Var}[\hat{s}]}$ across repeats')
ax.plot(summary.index, summary['mean_abs_error'], 's--', color='#2ca02c',
         label=r'$\mathrm{mean}|\hat{s} - s|$')
ax.set_xlabel('True numerosity')
ax.set_ylabel('Expected uncertainty')
ax.set_title('Decoding precision vs. stimulus')
ax.legend(fontsize=8)

fig.tight_layout()
plt.show()

# %%
# Interpretation
# -----------------------------------------------------------------
# * The **bias** curve (left) shows whether the model systematically
#   over- or under-estimates the true stimulus. A flat identity line
#   means no bias. Deviations often appear near the edges of the
#   stimulus range (the encoding model can't extrapolate beyond what
#   it was fit on).
# * The **expected uncertainty** curve (right) shows two flavours:
#   :math:`\sqrt{\mathrm{Var}[\hat{s}]}` across repeats (sampling
#   uncertainty of the posterior mean) and the mean absolute error
#   :math:`\mathrm{mean}|\hat{s} - s|`. They tend to track each other;
#   a gap means the estimator is biased (the MAE picks up bias, the SD
#   doesn't).
# * Under the Cramér–Rao bound the variance of an unbiased estimator
#   is at least :math:`1 / \mathcal{I}(s)`, so far from the boundary
#   :math:`\mathrm{Var}[\hat{s}] \approx 1 /` Fisher information.
#   :meth:`EncodingModel.get_expected_uncertainty` and
#   :meth:`EncodingModel.get_fisher_information` are the two ends of
#   that bound — see :doc:`/fisher_information` for the head-to-head.
# * Cost: simulate + decode :math:`\sim 10^3`–:math:`10^4` trials per
#   stimulus; use ``batch_stimuli=`` to bound memory for large grids.
