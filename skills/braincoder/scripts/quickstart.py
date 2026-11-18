#!/usr/bin/env python
"""A complete braincoder analysis on simulated data: copy this to start a new project.

    python quickstart.py

Fits a 1D tuning model to single-trial responses with leave-one-run-out
cross-validation, selects voxels inside each training fold, decodes the held-out run
and checks that the posteriors are calibrated. Takes about a minute on a laptop. To use
real data, replace load_data() and leave the rest.
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')   # must come before any keras import

import numpy as np
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.noise import LowRankNoise
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq
from braincoder.utils.calibration import calibration_summary


def load_data():
    """Return data (trials x voxels), paradigm (trials x 1) and the run of each trial."""
    rng = np.random.default_rng(0)
    n_voxels, n_runs, trials_per_run = 150, 6, 40
    n_trials = n_runs * trials_per_run
    stimulus = rng.uniform(5, 35, n_trials).astype(np.float32)
    index = pd.Index(np.arange(n_trials), name='trial')
    paradigm = pd.DataFrame({'stimulus': stimulus}, index=index)
    true = pd.DataFrame({'mu': rng.uniform(5, 35, n_voxels), 'sd': rng.uniform(4, 10, n_voxels),
                         'amplitude': 1., 'baseline': 0.}).astype(np.float32)
    tuned = rng.random(n_voxels) < .5                  # half of the voxels carry no signal
    true.loc[~tuned, 'amplitude'] = 1e-3
    data = GaussianPRF().simulate(paradigm, true, noise=1.)
    data.index = index
    return data, paradigm, np.repeat(np.arange(n_runs), trials_per_run)


def fit_tuning(data, paradigm):
    """Grid search, closed-form amplitude and baseline, then gradient descent."""
    fitter = ParameterFitter(GaussianPRF(), data, paradigm)
    grid = fitter.fit_grid(mu=np.linspace(5, 35, 16), sd=np.linspace(3, 15, 7),
                           amplitude=[1.], baseline=[0.], use_correlation_cost=True)
    init = fitter.refine_baseline_and_amplitude(grid)
    pars = fitter.fit(init_pars=init, max_n_iterations=1000, progressbar=False)
    return pars, fitter.r2


data, paradigm, runs = load_data()

# Check the inputs first: these two lines prevent the most common silent errors.
data = data.loc[:, data.notna().all() & (data.std() > 0)]
assert data.index.equals(paradigm.index), 'data and paradigm must share one index'

stimulus_grid = np.linspace(0, 40, 81).astype(np.float32)   # the flat prior for decoding
cv_r2, posteriors = [], []

for test_run in np.unique(runs):
    train, test = runs != test_run, runs == test_run

    pars, train_r2 = fit_tuning(data[train], paradigm[train])
    model = GaussianPRF()
    cv_r2.append(get_rsq(data[test], model.predict(paradigm[test], pars)))

    # Select voxels on training data only. A fixed top-50 keeps this example simple;
    # with many voxels, use fit_r2_mixture and r2_fdr_threshold instead.
    keep = train_r2.nlargest(50).index

    noise = LowRankNoise.fit(model, data.loc[train, keep], paradigm[train], pars.loc[keep],
                             stimulus_range=stimulus_grid, groups=runs[train])
    posteriors.append(model.get_stimulus_pdf(data.loc[test, keep], stimulus_grid,
                                             parameters=pars.loc[keep], noise_model=noise))

cv_r2 = pd.concat(cv_r2, axis=1).mean(axis=1)
posterior = pd.concat(posteriors).loc[data.index]

print(f'Mean cvR2 of the 50 best voxels: {cv_r2.nlargest(50).mean():.3f}')
print('Calibration of the held-out posteriors (coverage should match its level, pit_sd about 0.29):')
print(pd.Series(calibration_summary(posterior, paradigm['stimulus'].values)).round(3).to_string())
