"""
Fit the residual covariance matrix
=============================================

In this example, we fit a residual noise covariance matrix to
simulated data from a ``VonMisesPRF``-model.

"""

# We set up a simple VonMisesPRF model
from braincoder.models import VonMisesPRF
import numpy as np
import pandas as pd

# Set up six evenly spaced von Mises PRFs
centers = np.linspace(0.0, 2*np.pi, 6, endpoint=False)
parameters = pd.DataFrame({'mu':centers, 'kappa':1., 'amplitude':1.0, 'baseline':0.0},
                          index=pd.Index([f'Voxel {i+1}' for i in range(6)], name='voxel')).astype(np.float32)

# We have only 3 voxels, each with a linear combination of the 6 von Mises functions:
weights = np.array([[1, 0, 1],
                    [1, .5, 1],
                    [0, 1, 0],
                    [0, .5, 0],
                    [0, 0, 1],
                    [0, 0, 1]]).astype(np.float32)

model = VonMisesPRF(parameters=parameters, weights=weights)

# 50 random orientations
paradigm = np.random.rand(50) * np.pi*2

# Arbitrary covariance matrix
cov = np.array([[.5, 0.0, 0.0],
       [.25, .75, .25],
       [.25, .25, .75]])

data = model.simulate(noise=cov, weights=weights, paradigm=paradigm)

# %%

# Import ResidualFitter
from braincoder.optimize import ResidualFitter

fitter = ResidualFitter(model, data, paradigm, parameters, weights)

# omega is the covariance matrix, dof can be estimated when a 
# multivariate t-distribution (rather than a normal distribution)
# is used
omega, dof = fitter.fit(progressbar=False)
print(omega)

# %%