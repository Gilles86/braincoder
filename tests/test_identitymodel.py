from braincoder.decoders import WeightedEncodingModel
import numpy as np
from braincoder.utils import get_rsq

n_timepoints = 1000
n_dimensions = 2
n_voxels = 200000

paradigm = np.random.randn(n_timepoints, n_dimensions)
weights = np.random.randn(n_dimensions, n_voxels)

model = WeightedEncodingModel()

data = model.simulate(paradigm, weights, noise=.5)

costs, weights_, predictions = model.optimize(paradigm, data, ftol=1e-13)
r2 = get_rsq(data, predictions)

# ax = data.plot()
# predictions.plot(ax=ax, marker='+', lw=0)
