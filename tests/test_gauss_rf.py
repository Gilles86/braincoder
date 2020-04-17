from braincoder.models import VoxelwiseGaussianReceptiveFieldModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = VoxelwiseGaussianReceptiveFieldModel()


n_voxels = 15
n_timepoints =  50

paradigm = np.arange(0, 20)

parameters = np.ones((n_voxels, 4))
parameters[:, 0] = np.linspace(5, 15, n_voxels)
parameters[:, 3] = 0

data = model.simulate(paradigm, parameters, noise=0.1)

costs, pars_, pred_ =  model.fit_parameters(paradigm, data, progressbar=True)

predictions = model.get_predictions()
r2 = model.get_rsq(data)
r = model.get_r(data)

