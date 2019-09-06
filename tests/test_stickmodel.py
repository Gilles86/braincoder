from braincoder.models import StickModel
import numpy as np

n_voxels = 10
n_timepoints = 30
n_populations = 11
noise = 0.1

parameters = np.arange(n_populations)[:, np.newaxis]

weights = np.zeros((n_populations, n_voxels))
populations = np.random.choice(np.arange(1, 11), n_voxels)
intercepts = np.random.randn(n_voxels)

weights[0] = intercepts
weights[populations, np.arange(n_voxels)] = 1

paradigm = np.tile(np.arange(-1, 12), int(n_timepoints / 10))[:n_timepoints]

model = StickModel(parameters)
data = model.simulate(paradigm, weights=weights, noise=noise)
model.fit_weights(paradigm=paradigm, data=data)

r2 = model.get_rsq(data)

