"""
Decoding of stimuli from neural data
=============================================

Here we will simulate neural data given a ground truth encoding model
 and try to decode the stimulus from the data.
"""

# Set up a neural model
from braincoder.models import GaussianPRF
import numpy as np
import pandas as pd
import scipy.stats as ss

# Set up 100 random of PRF parameters
n = 20
n_trials = 50
noise = 1.

mu = np.random.rand(n) * 100
sd = np.random.rand(n) * 45 + 5
amplitude = np.random.rand(n) * 5
baseline = np.random.rand(n) * 2 - 1

parameters = pd.DataFrame({'mu':mu, 'sd':sd, 'amplitude':amplitude, 'baseline':baseline})

# We have a paradigm of random numbers between 0 and 100
paradigm = np.ceil(np.random.rand(n_trials) * 100)

model = GaussianPRF(parameters=parameters)
data = model.simulate(paradigm=paradigm, noise=noise)

# %%

# Now we fit back the PRF parameters
from braincoder.optimize import ParameterFitter, ResidualFitter
fitter = ParameterFitter(model, data, paradigm)
mu_grid = np.arange(0, 100, 5)
sd_grid = np.arange(5, 50, 5)

grid_pars = fitter.fit_grid(mu_grid, sd_grid, [1.0], [0.0], use_correlation_cost=True, progressbar=False)
grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

for par in ['mu', 'sd', 'amplitude', 'baseline']:
    print(f'Correlation grid-fitted parameter and ground truth for *{par}*: {ss.pearsonr(grid_pars[par], parameters[par])[0]:0.2f}')

gd_pars = fitter.fit(init_pars=grid_pars, progressbar=False)

for par in ['mu', 'sd', 'amplitude', 'baseline']:
    print(f'Correlation gradient descent-fitted parameter and ground truth for *{par}*: {ss.pearsonr(grid_pars[par], parameters[par])[0]:0.2f}')


# %%

# Now we fit the covariance matrix
stimulus_range = np.arange(1, 100).astype(np.float32)

model.init_pseudoWWT(stimulus_range=stimulus_range, parameters=gd_pars)
resid_fitter = ResidualFitter(model, data, paradigm, gd_pars)
omega, dof = resid_fitter.fit(progressbar=False)

# %%

# Now we simulate unseen test data:
test_paradigm = np.ceil(np.random.rand(n_trials) * 100)
test_data = model.simulate(paradigm=test_paradigm, noise=noise)

# And decode the test paradigm
posterior = model.get_stimulus_pdf(test_data, stimulus_range, model.parameters, omega=omega, dof=dof)

# %%

# Finally, we make some plots to see how well the decoder did
import matplotlib.pyplot as plt
import seaborn as sns

tmp = posterior.set_index(pd.Series(test_paradigm, name='ground truth'), append=True).loc[:8].stack().to_frame('p')

g = sns.FacetGrid(tmp.reset_index(), col='time', col_wrap=3)

g.map(plt.plot, 'stimulus', 'p', color='k')
# g.map(lambda data, **kwargs: plt.axvline(data, color='r'), 'ground truth')
def test(data, **kwargs):
    plt.axvline(data.mean(), c='k', ls='--', **kwargs)
g.map(test, 'ground truth')

# %%

# Let's look at the summary statistics of the posteriors posteriors
def get_posterior_stats(posterior):
    # Take integral over the posterior to get to the expectation (mean posterior)
    E = np.trapz(posterior*posterior.columns.values[np.newaxis,:], posterior.columns, axis=1)
    
    # Take the integral over the posterior to get the expectation of the distance to the 
    # mean posterior (i.e., standard deviation)
    sd = np.trapz(np.abs(E[:, np.newaxis] - posterior.columns.astype(float).values[np.newaxis, :]) * posterior, posterior.columns, axis=1)

    stats = pd.DataFrame({'E':E, 'sd':sd}, index=posterior.index)
    return stats

posterior_stats = get_posterior_stats(posterior)
plt.errorbar(test_paradigm, posterior_stats['E'],posterior_stats['sd'], fmt='o',)
plt.plot([0, 100], [0,100], c='k', ls='--')

plt.xlabel('Ground truth')
plt.ylabel('Mean posterior')

error = test_paradigm - posterior_stats['E']
error_abs = np.abs(error)
error_abs.name = 'error'

sns.lmplot(x='sd', y='error', data=posterior_stats.join(error_abs))

plt.xlabel('Standard deviation of posterior')
plt.ylabel('Objective error')

# %%