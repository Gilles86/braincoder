from braincoder.decoders import GaussianDecodingModel
from braincoder.generators import GaussianEncodingModel
from braincoder.utils import get_rsq
import numpy as np

paradigm = np.repeat(np.arange(-10, 11), 10)
n_voxels = 1000
enc_model = GaussianEncodingModel(np.linspace(-5, 6, n_voxels, endpoint=True), np.ones(n_voxels) * 3)

data = enc_model.simulate_data(paradigm, noise=.1)
data = data * 10

dec_model = GaussianDecodingModel(paradigm, data)

costs, pars, predictions = dec_model.optimize(min_nsteps=50000)

r2 = get_rsq(data, predictions)
