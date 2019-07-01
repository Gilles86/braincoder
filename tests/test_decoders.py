from braincoder.decoders import GaussianReceptiveFieldFitter
from braincoder.generators import GaussianEncodingModel
from braincoder.utils import get_rsq
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = GaussianReceptiveFieldFitter()
n_voxels = 1000

parameters = np.vstack((np.linspace(-10, 10, n_voxels), np.ones(n_voxels) * 5, np.ones(n_voxels) * 6, np.ones(n_voxels) * .2)).T
parameters = parameters.astype(np.float32)

paradigm = np.repeat(np.arange(-10, 11, 1), 1)
simulated_data = model.simulate(parameters, paradigm, noise=1)

dec_model = GaussianReceptiveFieldFitter()
costs, pars, predictions = model.optimize(paradigm.copy(), simulated_data,
        ftol=1e-12)

r2 = get_rsq(simulated_data, predictions)


plt.gcf()
plt.plot(paradigm, predictions)
plt.plot(paradigm, simulated_data, marker='+', lw=0, c='k')
plt.show()
