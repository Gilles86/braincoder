"""
Fit a 2D PRF model
=============================================

Here we fit a 2D PRF model to data from the Szinte (2024)-dataset.

"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

from braincoder.utils.data import load_szinte2024
import numpy as np
import pandas as pd

# Load data
data = load_szinte2024()
stimulus = data['stimulus']
grid_coordinates = data['grid_coordinates']

# Set up a function to draw a single frame
def update(frame):
    plt.clf()  # Clear the current figure
    plt.imshow(stimulus[frame, :, :].T, cmap='viridis')
    plt.title(f"Frame {frame}")

# Create the animation
fig = plt.figure()
ani = FuncAnimation(fig, update, frames=range(stimulus.shape[0]), interval=100)

# Convert to HTML for easy display
HTML(ani.to_html5_video())

# %%

# Set up a PRF model
# Now we will set up fake PRFs just to show how the data is structured
# We make a 9-by-9 grid of simulated PRFs
x, y = np.meshgrid(np.linspace(-6, 6, 3), np.linspace(-4, 4, 3))

# Set them up in a parameter table
# All PRFs have the same baseline and amplitude
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel

parameters = pd.DataFrame({'x':x.ravel(),
               'y':y.ravel(),
               'sd':2.5,
               'baseline':0.0,
               'amplitude':1.}).astype(np.float32)
model = GaussianPRF2DWithHRF(grid_coordinates=grid_coordinates, 
                      paradigm=stimulus,
                     parameters=parameters,
                    hrf_model=SPMHRFModel(tr=data['tr']))
# %%

# Let's plot all the RFs
rfs = model.get_rf(as_frame=True)

for i, rf in rfs.groupby(level=0):
    plt.subplot(3, 3, i+1)
    plt.title(f'RF {i+1}')
    plt.imshow(rf.unstack('y').loc[i].T)
    plt.axis('off')

# %%

# We simulate data for the given paradigm and parameters and plot the resulting time series
import seaborn as sns
data = model.simulate(noise=1.)
data.columns.set_names('voxel', inplace=True)
tmp = data.stack().to_frame('activity')
sns.relplot(x='frame', y='activity', data=tmp.reset_index(), hue='voxel', kind='line', palette=sns.color_palette('tab10', n_colors=parameters.shape[0]), aspect=2.)

# %%

# We can also fit parameters back to data
from braincoder.optimize import ParameterFitter

# We set up a parameter fitter
par_fitter = ParameterFitter(model, data, stimulus)

# We set up a grid of parameters to search over
x = np.linspace(-8, 8, 20)
y = np.linspace(-4, 4, 20)
sd = np.linspace(1, 5, 10)

# For now, we only use one amplitude and baseline, because we
# use a correlation cost function, which is indifferent to
# the overall scaling of the model
# We can easily estimate these later using OLS
amplitudes = [1.0]
baseline = [0.0]

# Note that the grids should be given in the correct order (can be found back in
# model.parameter_labels)
grid_pars = par_fitter.fit_grid(x, y, sd, baseline, amplitudes, use_correlation_cost=True)

# Once we have the best parameters from the grid, we can optimize the baseline
# and amplitude
refined_grid_pars = par_fitter.refine_baseline_and_amplitude(grid_pars)

# We get the explained variance of these parameters
from braincoder.utils import get_rsq
refined_grid_r2 = get_rsq(data, model.predict(parameters=refined_grid_pars))

# Now we use gradient descent to further optimize the parameters
pars = par_fitter.fit(init_pars=refined_grid_pars, learning_rate=1e-2, max_n_iterations=5000,
        min_n_iterations=100,
        r2_atol=0.0001)

fitted_r2 = get_rsq(data, model.predict(parameters=pars))

# The fitted R2s tend to be a bit better than the grid R2s
display(refined_grid_r2.to_frame('r2').join(fitted_r2.to_frame('r2'), lsuffix='_grid', rsuffix='_fitted'))

# The real parameters are very similar to the estimated parameters
display(pars.join(parameters, lsuffix='_fit', rsuffix='_true'))

# %%

# Decode the *stimulus* from "unseen" data:
# First we need to fit a noise model
from braincoder.optimize import ResidualFitter
resid_fitter = ResidualFitter(model, data, stimulus, parameters=pars)
omega, dof = resid_fitter.fit()

# Simulate new "unseen" data
unseen_data = model.simulate(noise=1.)

# For stimulus reconstruction, we slightly downsample the stimulus space
# otherwise the optimization takes too long on a CPU
# we can do that by simply setting up a new model with a different grid
data = load_szinte2024(resize_factor=2.5)
grid_coordinates = data['grid_coordinates']
stimulus = data['stimulus']

model = GaussianPRF2DWithHRF(grid_coordinates=grid_coordinates, 
                     parameters=parameters,
                    hrf_model=SPMHRFModel(tr=data['tr']))

# We set up a stimulus fitter
from braincoder.optimize import StimulusFitter
stim_fitter = StimulusFitter(unseen_data, model, omega)

# Legacy Adam is a bit faster than the default Adam optimizer on M1
# Learning rate of 1.0 is a bit high, but works well here
reconstructed_stimulus = stim_fitter.fit(legacy_adam=True, min_n_iterations=200, max_n_iterations=200, learning_rate=.1)

# Here we make a movie of the decoded stimulus
# Set up a function to draw a single frame
vmin, vmax = 0.0, np.quantile(reconstructed_stimulus.values.ravel(), 0.95)

def update(frame):
    plt.clf()  # Clear the current figure
    plt.imshow(reconstructed_stimulus.stack('y').loc[frame], cmap='viridis', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(f"Frame {frame}")

# Create the animation
fig = plt.figure()
ani = FuncAnimation(fig, update, frames=range(stimulus.shape[0]), interval=100)

HTML(ani.to_html5_video())