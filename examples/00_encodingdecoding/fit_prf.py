"""
Different flavors of visual population receptive field models
==============================================================

In this example script we will try out increasingly complex models for
visual population receptive fields (PRFs). We will start with a simple
Gaussian PRF model, and then add more complexity step by step.

"""

# %%
# Load data
# ---------
# First we load in the data. We will use the Szinte (2024)-dataset.
from braincoder.utils.data import load_szinte2024
import matplotlib.pyplot as plt

data = load_szinte2024()

# This is the visual stimulus ("design matrix")
paradigm = data['stimulus']
grid_coordinates = data['grid_coordinates']

# This is the fMRI response data
d = data['v1_timeseries']
d.index.name = 'frame'
tr = data['tr']


# %%
# Simple 2D Gaussian Recetive Field model
# -------------------------------------
# Now we set up a simple Gaussian PRF model
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
hrf_model = SPMHRFModel(tr=tr)
model_gauss = GaussianPRF2DWithHRF(data=d, paradigm=paradigm, hrf_model=hrf_model, grid_coordinates=grid_coordinates)

# %%
# And a parameter fitter...
from braincoder.optimize import ParameterFitter
par_fitter = ParameterFitter(model=model_gauss, data=d, paradigm=paradigm)


# %%
# Now we try out a relatively coarse grid search to find the some
# parameters to start the gradient descent from.

import numpy as np
x = np.linspace(-8, 8, 10)
y = np.linspace(-4, 4, 10)
sd = np.linspace(0.1, 4, 10)

# We start the grid search using a correlation cost, so ampltiude
# and baseline do not influence those results.
# We will optimize them later using OLS.
baseline = [0.0]
amplitude = [1.0]

# Now we can do the grid search
pars_gauss_grid = par_fitter.fit_grid(x, y, sd, baseline, amplitude, correlation_cost=True)

# And refine the baseline and amplitude parameters using OLS
pars_gauss_ols = par_fitter.refine_baseline_and_amplitude(pars_gauss_grid)


# %%
# Here we can plot the resulting r2s of the grid search
r2_gauss_ols = par_fitter.get_rsq(pars_gauss_ols)

import seaborn as sns
sns.kdeplot(r2_gauss_ols, shade=True)
sns.despine()
# %%

# %%
# We can substantially improve the fit by using gradient descent optimisation
pars_gauss_gd = par_fitter.fit(init_pars=pars_gauss_ols, max_n_iterations=1000)

# %%
r2_gauss_gd = par_fitter.get_rsq(pars_gauss_gd)
sns.kdeplot(r2_gauss_gd, shade=True)

import pandas as pd
r2 = pd.concat((r2_gauss_ols, r2_gauss_gd), keys=['r2_ols', 'r2_gd'], axis=1)

# %%
# Clearly, the gradient descent optimization improves the fit substantially.
sns.relplot(x='r2_ols', y='r2_gd', data=r2.reset_index(), kind='scatter')
plt.plot([0, 1], [0, 1], 'k--')
#
#  %%

# %%
# Fit HRFs
# --------
# The standard canonical (SPM) HRF we use is often not a great fit to actual
# data. To better account for the HRF. We can optimize the HRFs per voxel.
# We first initialize a GaussianPRF-model with a flexible HRF.
model_hrf = GaussianPRF2DWithHRF(data=d, paradigm=paradigm, hrf_model=hrf_model,
                             grid_coordinates=grid_coordinates, flexible_hrf_parameters=True)

par_fitter_hrf = ParameterFitter(model=model_hrf, data=d, paradigm=paradigm)

# We set hrf_delay and hrf_dispersion to standard values
pars_gauss_gd['hrf_delay'] = 6
pars_gauss_gd['hrf_dispersion'] = 1

pars_gauss_hrf = par_fitter_hrf.fit(init_pars=pars_gauss_gd, max_n_iterations=1000)

# %%
r2_gauss_hrf = par_fitter_hrf.get_rsq(pars_gauss_hrf)

r2 = pd.concat((r2_gauss_gd, r2_gauss_hrf), keys=['r2_gd', 'r2_hrf'], axis=1)
sns.relplot(x='r2_gd', y='r2_hrf', data=r2.reset_index(), kind='scatter')
plt.plot([0, 1], [0, 1], 'k--')

# %%
# Here we plot the predicted time courses of the original model
# and the model with the optimized HRFs for 9 voxels where the fit
# improved the most. You can clearly see that, in general, the
# HRFs have shorter delays than the default setting.
improvement = r2_gauss_hrf - r2_gauss_gd
largest_improvements = improvement.sort_values(ascending=False).index[:9]
pred_gauss_gd = model_gauss.predict(parameters=pars_gauss_gd)
pred_gauss_hrf = model_hrf.predict(parameters=pars_gauss_hrf)
pred = pd.concat((d.loc[:, largest_improvements], pred_gauss_gd.loc[:, largest_improvements], pred_gauss_hrf.loc[:, largest_improvements]), axis=1, keys=['data', 'gauss', 'gauss+hrf'], names=['model'])

#
tmp = pred.stack(['model', 'source']).to_frame('value')
sns.relplot(x='frame', y='value', hue='model', col='source', data=tmp.reset_index(), kind='line', col_wrap=3)


# %%

# %%
# Fit a Difference of Gaussians model
# -----------------------------------
# Now we will try to fit a Difference of Gaussians model. This model
# has two Gaussian receptive fields, one excitatory and one inhibitory.
# The inhibitory receptive field is subtracted from the excitatory one.
# The resulting receptive field is then convolved with the HRF.
from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
model_dog = DifferenceOfGaussiansPRF2DWithHRF(data=d, paradigm=paradigm, hrf_model=hrf_model,
                                         grid_coordinates=grid_coordinates, flexible_hrf_parameters=True)

pars_dog_init = pars_gauss_hrf.copy()
# This is the relative amplitude of the inhibitory receptive field
# compared to the excitatory one.
pars_dog_init['srf_amplitude'] = 0.1

# This is the relative size of the inhibitory receptive field
# compared to the excitatory one.
pars_dog_init['srf_size'] = 2.

# Let's set up a new parameterfitter 
par_fitter_dog = ParameterFitter(model=model_dog, data=d, paradigm=paradigm)

# Note how, for now, we are not optimizing the HRF parameters.
pars_dog = par_fitter_dog.fit(init_pars=pars_dog_init, max_n_iterations=1000,
                              fixed_pars=['hrf_delay', 'hrf_dispersion'])

# Now we optimize _with_ the HRF parameters
pars_dog_hrf = par_fitter_dog.fit(init_pars=pars_dog, max_n_iterations=1000)

r2_dog_hrf = par_fitter_dog.get_rsq(pars_dog_hrf)

sns.relplot(x='r2_hrf', y='r2_dog_hrf', data=pd.concat((r2_gauss_hrf, r2_dog_hrf), axis=1,
                                                       keys=['r2_hrf', 'r2_dog_hrf']).reset_index(), kind='scatter')
# %%


# %%
# Here, we plot the predicted time courses of the difference-of-gaussians
# model versus the original Gaussian model for the 9 voxels where the fit
# imoproved the most.
improvement = r2_dog_hrf - r2_gauss_hrf
largest_improvements = improvement.sort_values(ascending=False).index[:9]
pred_dog_hrf = model_dog.predict(parameters=pars_dog_hrf)
pred = pd.concat((d.loc[:, largest_improvements], pred_gauss_hrf.loc[:, largest_improvements], pred_dog_hrf.loc[:, largest_improvements]), axis=1, keys=['data', 'gauss+hrf', 'dog+hrf'], names=['model'])

tmp = pred.stack(['model', 'source']).to_frame('value')
sns.relplot(x='frame', y='value', hue='model', col='source', data=tmp.reset_index(), kind='line', col_wrap=3,
            palette=['k'] + sns.color_palette(),
            hue_order=['data', 'gauss+hrf', 'dog+hrf'])


# %%
# Divisve Normalization PRF model
# -------------------------------
# The most complex model we have is the DN-PRF model (Aqil et al., 2021).
# This model has a Gaussian excitatory receptive field, and a Gaussian
# inhibitory receptive field. The excitatory receptive field is divided
# by the sum of the excitatory and inhibitory receptive fields. 
# The resulting receptive field is then convolved with the HRF.
from braincoder.models import DivisiveNormalizationGaussianPRF2DWithHRF
model_dn = DivisiveNormalizationGaussianPRF2DWithHRF(data=d,
                                              paradigm=paradigm,
                                              hrf_model=hrf_model,
                                              grid_coordinates=grid_coordinates,
                                              flexible_hrf_parameters=True)

pars_dn_init = pars_dog_hrf.copy()
pars_dn_init['srf_amplitude'] = 0.01
pars_dn_init['srf_size'] = 2.
pars_dn_init['baseline'] = 0.0
pars_dn_init['neural_baseline'] = 1.0
pars_dn_init['surround_baseline'] = 1.0

par_fitter_dn = ParameterFitter(model=model_dn, data=d, paradigm=paradigm)
# Without HRF
pars_dn = par_fitter_dn.fit(init_pars=pars_dn_init, max_n_iterations=1000, fixed_pars=['hrf_delay', 'hrf_dispersion'])

# With HRF
pars_dn = par_fitter_dn.fit(init_pars=pars_dn, max_n_iterations=1000)

# %%
# Again, let's  plot the R2 improvements
r2_dn = par_fitter_dn.get_rsq(pars_dn)
sns.relplot(x='r2_dog_hrf', y='r2_dn', data=pd.concat((r2_dog_hrf, r2_dn), axis=1,
                                                       keys=['r2_dog_hrf', 'r2_dn']).reset_index(), kind='scatter')

plt.plot([0, 1], [0, 1], 'k--')

# %%
improvement = r2_dn - r2_dog_hrf
largest_improvements = improvement.sort_values(ascending=False).index[:9]

pred_dn = model_dn.predict(parameters=pars_dn)
pred = pd.concat((d.loc[:, largest_improvements], pred_dog_hrf.loc[:, largest_improvements], pred_dn.loc[:, largest_improvements]), axis=1, keys=['data', 'dog+hrf', 'dn+hrf'], names=['model'])

tmp = pred.stack(['model', 'source']).to_frame('value')
sns.relplot(x='frame', y='value', hue='model', col='source', data=tmp.reset_index(), kind='line', col_wrap=3,
            palette=['k'] + sns.color_palette(),
            hue_order=['data', 'dog+hrf', 'dn+hrf'])
# %%


# Decoding
# --------
# We can also use the fitted models to decode the stimulus from the
# fMRI response. Let's compare our simplest model versus our most
# complex model.

# First we fit the noise models
from braincoder.optimize import ResidualFitter, StimulusFitter

# Let's first get grid coordinates and paradigm at a slightly lower resolution
data = load_szinte2024(resize_factor=2.5)
grid_coordinates = data['grid_coordinates']
paradigm = data['stimulus']

best_voxels_gauss = r2_gauss_gd[pars_gauss_gd['sd'] > 0.5].sort_values(ascending=False).index[:200]

model_gauss = GaussianPRF2DWithHRF(data=d[best_voxels_gauss],
                                   hrf_model=hrf_model,
                                   grid_coordinates=grid_coordinates.astype(np.float32),
                                   parameters=pars_gauss_gd.loc[best_voxels_gauss].astype(np.float32))

resid_fitter_gauss = ResidualFitter(model=model_gauss, data=d.loc[:, best_voxels_gauss],
                                    paradigm=paradigm.astype(np.float32), parameters=pars_gauss_gd.loc[best_voxels_gauss].astype(np.float32))
omega_gauss, _ = resid_fitter_gauss.fit()




# %%


# %%
best_voxels_dn = r2_dn[pars_dn['sd'] > 0.5].sort_values(ascending=False).index[:200]

model_dn = DivisiveNormalizationGaussianPRF2DWithHRF(data=d[best_voxels_dn], 
                                              hrf_model=hrf_model,
                                              grid_coordinates=grid_coordinates.astype(np.float32),
                                              parameters=pars_dn.loc[best_voxels_dn].astype(np.float32))

resid_fitter_dn = ResidualFitter(model=model_dn, data=d.loc[:, best_voxels_dn],
                                    paradigm=paradigm, parameters=pars_dn.loc[best_voxels_dn])

omega_dn, _ = resid_fitter_dn.fit()

# %%
# Decoded stimulus: Gaussian model
# ===============================
# Now we can decode the stimulus from the fMRI responses
stim_fitter_gauss = StimulusFitter(model=model_gauss, data=d.loc[:, best_voxels_gauss], omega=omega_gauss)
stim_gauss = stim_fitter_gauss.fit(l2_norm=0.01, learning_rate=0.01, max_n_iterations=1000)

# %%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def play_reconstruction(reconstructed_stimulus):

  # Here we make a movie of the decoded stimulus
  # Set up a function to draw a single frame
  vmin, vmax = 0.0, np.quantile(reconstructed_stimulus.values.ravel(), 0.99)

  def update(frame):
      plt.clf()  # Clear the current figure
      plt.imshow(reconstructed_stimulus.stack('y').loc[frame].iloc[::-1, :], cmap='viridis', vmin=vmin, vmax=vmax)
      plt.axis('off')
      plt.title(f"Frame {frame}")

  # Create the animation
  fig = plt.figure()
  ani = FuncAnimation(fig, update, frames=range(paradigm.shape[0]), interval=100)

  return HTML(ani.to_html5_video())

play_reconstruction(stim_gauss)


# %%
# Decoded stimulus: DN model
# ==========================
stim_fitter_dn = StimulusFitter(model=model_dn, data=d.loc[:, best_voxels_dn], omega=omega_dn)
stim_dn = stim_fitter_dn.fit(l2_norm=0.01, learning_rate=0.01, max_n_iterations=1000)

# %%
play_reconstruction(stim_dn)
# As you can see, the DN model works a lot better than the Gaussian model. ;)

# %%
