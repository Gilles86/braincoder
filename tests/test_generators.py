from braincoder.generators import GaussianEncodingModel
import numpy as np

model = GaussianEncodingModel(np.arange(1, 11), np.ones(10))

data = model.get_response_profile(np.arange(4, 7))
