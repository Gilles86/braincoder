from braincoder.models import StickModel
import numpy as np

n_voxels = 5
n_timepoints = 30
n_populations = 11

parameters = np.arange(n_populations)[:, np.newaxis]

weights = np.zeros((n_populations, n_voxels))
populations = np.random.choice(np.arange(0, 11), 5)
weights[populations, np.arange(n_voxels)] = 1

weights = np.random.randn(n_populations, n_voxels)
paradigm = np.tile(np.arange(-1, 12), int(n_timepoints / 10))[:n_timepoints]

model = StickModel(parameters)
data = model.simulate(paradigm, weights=weights, noise=0.1)
model.fit_weights(paradigm=paradigm, data=data)

r2 = model.get_rsq(data)

