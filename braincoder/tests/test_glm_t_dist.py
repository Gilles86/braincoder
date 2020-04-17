from braincoder.models import EncodingModel, GLMModel

# from braincoder.decoders import WeightedEncodingModel
import numpy as np
from braincoder.utils import get_rsq, get_r
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as ss

n_timepoints = 100
n_dimensions = 2
n_voxels = 25

paradigm = np.random.randn(n_timepoints, n_dimensions)
weights = np.random.randn(n_dimensions, n_voxels)

model = GLMModel()

data = model.simulate(paradigm, weights=weights, noise=0.0)
data2 = model.simulate(paradigm, weights=weights, noise=0.0)

data += ss.t(3).rvs(data.shape) * .5 
data2 += ss.t(3).rvs(data2.shape) * .5

test = model.fit_weights(paradigm, data, l2_cost=0.0)

predictions = model.get_predictions()
r2 = model.get_rsq(data)
r22 = model.get_rsq(data2)
r = model.get_r(data)

model.fit_residuals(paradigm, data, residual_dist='t')
stimulus_range = np.linspace(-5, 5, 100, dtype=np.float32)[:, np.newaxis]
# stimulus_range = np.repeat(stimulus_range, n_dimensions, 1)

stimulus_range = np.array(np.meshgrid(*[np.linspace(-3, 3, 20) for i in range(n_dimensions)]))

stimulus_range = stimulus_range.reshape((n_dimensions, np.prod(stimulus_range.shape[1:]))).T

p, map_, sd, ci = model.get_stimulus_posterior(data, stimulus_range)
r_decode = get_r(map_, paradigm)

p, map_, sd, ci = model.get_stimulus_posterior(data2, stimulus_range)
r_decode2 = get_r(map_, paradigm)
