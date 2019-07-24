from braincoder.decoders import WeightedEncodingModel
import numpy as np
from braincoder.utils import get_rsq
import matplotlib.pyplot as plt
import seaborn as sns

n_timepoints = 100
n_dimensions = 5
n_voxels = 25

paradigm = np.random.randn(n_timepoints, n_dimensions)
weights = np.random.randn(n_dimensions, n_voxels)

model = WeightedEncodingModel()

noise = np.arange(n_voxels) / n_voxels * 5.
noise = 0.0
data = model.simulate(paradigm, weights, noise=1.0)

data += np.random.randn(len(data), 1).astype(np.float32) * 1

costs = model.fit(paradigm, data)

predictions = model.get_predictions()
r2 = model.get_rsq(data)
r = model.get_r(data)

# data2 = model.simulate(paradigm, weights, noise=1.)
# r22 = model.get_rsq(data2)

# sns.distplot(r2)
# sns.distplot(r22)

# plt.plot(data.loc[:, :2])
# plt.plot(predictions.loc[:, :2], ls='--')
# plt.xlim(0, 10)
plt.plot(costs)
plt.show()

# costs, weights_, predictions = model.optimize(paradigm, data, ftol=1e-13)
# r2 = get_rsq(data, predictions)

# ax = data.plot()
# predictions.plot(ax=ax, marker='+', lw=0)
