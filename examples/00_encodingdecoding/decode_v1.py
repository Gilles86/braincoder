"""
Recontruct a 2D visual stimulus from real fMRI data
===================================================
Bla.
Here we decode a 2D stimulus from te Szinte (2024)-dataset.

"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns
from braincoder.utils.data import load_szinte2024
import numpy as np
import pandas as pd

# Load data (high-res for PRF fitting)
data = load_szinte2024()
stimulus = data['stimulus']
grid_coordinates = data['grid_coordinates']
tr = data['tr']
data = data['v1_timeseries']

from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel

model = GaussianPRF2DWithHRF(grid_coordinates=grid_coordinates, 
                      paradigm=stimulus,
                    hrf_model=SPMHRFModel(tr=tr))
# %%

"""
Fit parameters
--------------
"""
# Fit PRF parameters (encoding model)
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

# Now we use gradient descent to further optimize the parameters
pars = par_fitter.fit(init_pars=refined_grid_pars, learning_rate=1e-2, max_n_iterations=5000,
        min_n_iterations=100,
        r2_atol=0.0001)

from braincoder.utils import get_rsq
fitted_r2 = get_rsq(data, model.predict(parameters=pars))

# %%

"""
Analyze PRF locations
---------------------
"""
# Let's plot the location of all these PRFs:
sns.relplot(x='x', y='y', hue='r2', data=pars.join(fitted_r2.to_frame('r2')), size='sd', sizes=(10, 100), palette='viridis')
plt.title('PRF locations')

# Now we get the 250 best voxels:
best_voxels = fitted_r2.sort_values(ascending=False).iloc[:250].index

plt.figure()
sns.relplot(x='x', y='y', hue='r2', data=pars.loc[best_voxels].join(fitted_r2.to_frame('r2')), size='sd', sizes=(10, 100), palette='viridis')
plt.title('PRF locations (best 250 voxels)')

"""
Fit noise model on residuals
----------------------------
"""
from braincoder.optimize import ResidualFitter
resid_fitter = ResidualFitter(model, data.loc[:, best_voxels], stimulus, parameters=pars.loc[best_voxels])
omega, dof = resid_fitter.fit()

"""
Reconstruct stimulus
--------------------

For stimulus reconstruction, we slightly downsample the stimulus space
otherwise the optimization takes too long on a CPU
we can do that by simply setting up a new model with a different grid

"""
data = load_szinte2024(resize_factor=2.5)
grid_coordinates = data['grid_coordinates']
stimulus = data['stimulus']
tr = data['tr']
data = data['v1_timeseries']

model = GaussianPRF2DWithHRF(grid_coordinates=grid_coordinates, 
                     parameters=pars.loc[best_voxels],
                    hrf_model=SPMHRFModel(tr=tr))

# We set up a stimulus fitter
from braincoder.optimize import StimulusFitter
stim_fitter = StimulusFitter(data.loc[:, best_voxels], model, omega)

# Legacy Adam is a bit faster than the default Adam optimizer on M1
# Learning rate of 1.0 is a bit high, but works well here
reconstructed_stimulus = stim_fitter.fit(legacy_adam=True, min_n_iterations=200, max_n_iterations=500, learning_rate=.1)

def play_reconstruction(reconstructed_stimulus):

  # Here we make a movie of the decoded stimulus
  # Set up a function to draw a single frame
  vmin, vmax = 0.0, np.quantile(reconstructed_stimulus.values.ravel(), 0.99)

  def update(frame):
      plt.clf()  # Clear the current figure
      plt.imshow(reconstructed_stimulus.stack('y').loc[frame], cmap='viridis', vmin=vmin, vmax=vmax)
      plt.axis('off')
      plt.title(f"Frame {frame}")

  # Create the animation
  fig = plt.figure()
  ani = FuncAnimation(fig, update, frames=range(stimulus.shape[0]), interval=100)

  return HTML(ani.to_html5_video())

play_reconstruction(reconstructed_stimulus)

# %%

"""
Reconstruct stimulus with L2-norm
---------------------------------
Note how this reconstructed stimulus is very sparse and doesn't look a lot like
the actual image. Part of the problem is that
the optimisation is very unconstrained: we have 250 voxels times 
150 (correlated!) datapoints, but ~800 time 150 stimulus pixels
We can induce less extreme intensities, and thereby less 
sparseness, by inducing a L2-penalty on the stimulus intensities
"""

reconstructed_stimulus = stim_fitter.fit(legacy_adam=True, min_n_iterations=200, max_n_iterations=1000, learning_rate=0.1, l2_norm=0.01)

play_reconstruction(reconstructed_stimulus)

# %%


"""
Reconstruct stimulus with L1-norm
---------------------------------
For completeness, one can also use a sparse-inducing L1-norm
"""
reconstructed_stimulus = stim_fitter.fit(legacy_adam=True, min_n_iterations=200, max_n_iterations=1000, learning_rate=.1, l1_norm=0.01)
play_reconstruction(reconstructed_stimulus)