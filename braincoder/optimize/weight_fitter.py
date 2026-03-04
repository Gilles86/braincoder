import pandas as pd
import numpy as np
from tensorflow.linalg import lstsq
from ..utils import format_data, format_parameters
from ..models import LinearModelWithBaseline


class WeightFitter(object):
    """Closed-form solver for voxel weights given fixed parameters.

    Uses TensorFlow's ``lstsq`` to compute weights that best map basis predictions
    to measured data, optionally with L2 regularization via ``alpha``.
    """

    def __init__(self, model, parameters, data, paradigm):
        self.model = model
        self.parameters = format_parameters(parameters)
        self.data = format_data(data)
        self.paradigm = self.model.get_paradigm(paradigm)

    def fit(self, alpha=0.0):
        """Solve for weights using least squares with optional L2 regularization."""
        parameters = self.model._get_parameters(self.parameters)
        parameters_ = parameters.values[np.newaxis, ...] if parameters is not None else None

        basis_predictions = self.model._basis_predictions(self.paradigm.values[np.newaxis, ...], parameters_)

        weights = lstsq(basis_predictions, self.data.values, l2_regularizer=alpha)[0]

        if (parameters is None) or type(self.model) == LinearModelWithBaseline:
            weights = pd.DataFrame(weights.numpy(),
                               columns=self.data.columns)
        else:
            weights = pd.DataFrame(weights.numpy(), index=self.parameters.index,
                               columns=self.data.columns)

        return weights
