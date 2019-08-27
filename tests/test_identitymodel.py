from braincoder.models import EncodingModel, IdentityModel

# from braincoder.decoders import WeightedEncodingModel
import numpy as np
from braincoder.utils import get_rsq
import matplotlib.pyplot as plt
import seaborn as sns

n_timepoints = 100
n_dimensions = 5
n_voxels = 150

paradigm = np.random.randn(n_timepoints, n_dimensions)
weights = np.random.randn(n_dimensions, n_voxels)

model = IdentityModel()

data = model.simulate(paradigm, weights=weights, noise=1.0)
test = model.fit_weights(paradigm, data, l2_cost=0.0)
data2 = model.simulate(paradigm, weights=weights, noise=1.0)

predictions = model.get_predictions()
r2 = model.get_rsq(data)
r22 = model.get_rsq(data2)
r = model.get_r(data)
