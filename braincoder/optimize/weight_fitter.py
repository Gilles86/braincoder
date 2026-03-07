import pandas as pd
import numpy as np
import keras
from keras import ops
from ..utils import format_data, format_parameters
from ..models import LinearModelWithBaseline


class WeightFitter(object):
    """Closed-form solver for voxel weights given fixed parameters.

    Uses least-squares to compute weights that best map basis predictions
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

        A = basis_predictions[0]  # (n_timepoints, n_basis)
        b = self.data.values       # (n_timepoints, n_voxels)

        if alpha > 0.0:
            # Tikhonov-regularized least squares: (A^T A + alpha*I) x = A^T b
            n_basis = A.shape[-1] if A.shape[-1] is not None else ops.shape(A)[-1]
            AtA = ops.matmul(ops.transpose(A), A) + alpha * ops.eye(int(n_basis))
            Atb = ops.matmul(ops.transpose(A), ops.convert_to_tensor(b, dtype='float32'))
            import tensorflow as tf
            weights_vals = tf.linalg.solve(AtA, Atb)
        else:
            weights_vals = ops.lstsq(A, ops.convert_to_tensor(b, dtype='float32'))

        if hasattr(weights_vals, 'numpy'):
            weights_vals = weights_vals.numpy()

        if (parameters is None) or type(self.model) == LinearModelWithBaseline:
            weights = pd.DataFrame(weights_vals,
                               columns=self.data.columns)
        else:
            weights = pd.DataFrame(weights_vals, index=self.parameters.index,
                               columns=self.data.columns)

        return weights
