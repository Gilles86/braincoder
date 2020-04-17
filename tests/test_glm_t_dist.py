from braincoder.models import EncodingModel, GLMModel

# from braincoder.decoders import WeightedEncodingModel
import numpy as np
from braincoder.utils import get_rsq, get_r
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as ss

n_timepoints = 100
n_dimensions = 1
n_voxels = 15

paradigm = np.random.randn(n_timepoints, n_dimensions)
weights = np.random.randn(n_dimensions, n_voxels)

model = GLMModel()
model_t = GLMModel()

data = model.simulate(paradigm, weights=weights, noise=0.0)

data += ss.t(3).rvs(data.shape) * 3

model.fit_weights(paradigm, data, l2_cost=0.0)
model_t.fit_weights(paradigm, data, l2_cost=0.0)

predictions = model.get_predictions()
r2 = model.get_rsq(data)
r = model.get_r(data)

model.fit_residuals(paradigm, data, residual_dist='gaussian', also_fit_weights=True)
model_t.fit_residuals(paradigm, data, residual_dist='t', also_fit_weights=True)

stimulus_range = np.array(np.meshgrid(*[np.linspace(-3, 3, 200) for i in range(n_dimensions)]))
stimulus_range = stimulus_range.reshape((n_dimensions, np.prod(stimulus_range.shape[1:]))).T

p, map_, sd, ci = model.get_stimulus_posterior(data, stimulus_range)
p_t, map_t, sd_t, ci_t = model_t.get_stimulus_posterior(data, stimulus_range)


in_ci = (paradigm > ci[0]) & (paradigm < ci[1])
in_ci_t = (paradigm > ci_t[0]) & (paradigm < ci_t[1])

