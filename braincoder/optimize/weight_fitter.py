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

    def fit(self, alpha=0.0, fit_intercept=False):
        """Solve for weights using least squares with optional L2 regularization.

        ``fit_intercept=True`` prepends a column of ones to the basis design
        matrix and solves for a per-voxel intercept *jointly* with the
        weights, in the same closed-form regression. This is the general fix
        for basis models (e.g. ``LinearModelWithBaseline``, or any model
        whose ``_basis_predictions`` ignores ``parameters``) that declare a
        baseline/offset without folding it into the fit: without an
        intercept term, ``WeightFitter`` regresses data on the raw basis
        *through the origin*, which only recovers the correct slope when the
        basis already happens to be mean-zero — otherwise the fitted weight
        is biased by whatever DC level the basis has. Returns
        ``(weights, intercept)`` instead of plain ``weights`` when enabled.
        The intercept column is excluded from the L2 penalty when
        ``alpha>0`` (standard ridge convention — the offset isn't shrunk).

        Mutually exclusive with the ``LinearModelWithBaseline`` fixed-known-
        baseline path below (that path already subtracts a supplied
        ``parameters['baseline']`` from the data before the regression;
        jointly fitting an intercept on top of an already-baseline-removed
        target is redundant / ill-posed). Pass ``parameters=None`` when
        constructing the fitter if you want ``fit_intercept`` to estimate
        the baseline itself instead of supplying one.
        """
        parameters = self.model._get_parameters(self.parameters)
        parameters_ = parameters.values[np.newaxis, ...] if parameters is not None else None

        basis_predictions = self.model._basis_predictions(self.paradigm.values[np.newaxis, ...], parameters_)

        A = basis_predictions[0]  # (n_timepoints, n_basis)
        b = self.data.values       # (n_timepoints, n_voxels)

        # LinearModelWithBaseline predicts basis@weights + baseline; the baseline
        # is a fixed known parameter, so regress the baseline-removed data.
        # Otherwise the all-positive value feature absorbs each voxel's DC offset
        # and the inverted (decoding) posterior is flat in the candidate stimulus
        # (decodes at r≈0). Gated to this model; GaussianPRF/LinearModel untouched.
        has_fixed_baseline = ((parameters is not None)
                               and (type(self.model) == LinearModelWithBaseline))
        if has_fixed_baseline:
            b = b - parameters['baseline'].values[np.newaxis, :]

        if fit_intercept:
            if has_fixed_baseline:
                raise ValueError(
                    "fit_intercept=True is redundant with LinearModelWithBaseline's "
                    "own fixed-baseline handling (parameters['baseline'] is already "
                    "subtracted above) — either pass parameters=None so "
                    "fit_intercept estimates the baseline itself, or keep "
                    "fit_intercept=False and supply the baseline yourself.")
            A_t = ops.convert_to_tensor(A, dtype='float32')
            ones = ops.ones((ops.shape(A_t)[0], 1), dtype='float32')
            A = ops.concatenate([ones, A_t], axis=1)

        if alpha > 0.0:
            # Tikhonov-regularized least squares: (A^T A + alpha*I) x = A^T b
            n_basis = A.shape[-1] if A.shape[-1] is not None else ops.shape(A)[-1]
            n_basis = int(n_basis)
            reg_diag = np.ones(n_basis, dtype=np.float32)
            if fit_intercept:
                reg_diag[0] = 0.0    # don't shrink the intercept
            reg = alpha * ops.diag(ops.convert_to_tensor(reg_diag))
            AtA = ops.matmul(ops.transpose(A), A) + reg
            Atb = ops.matmul(ops.transpose(A), ops.convert_to_tensor(b, dtype='float32'))
            weights_vals = ops.solve(AtA, Atb)
        else:
            weights_vals = ops.lstsq(A, ops.convert_to_tensor(b, dtype='float32'))

        weights_vals = ops.convert_to_numpy(weights_vals)

        intercept_vals = None
        if fit_intercept:
            intercept_vals = weights_vals[0]
            weights_vals = weights_vals[1:]

        if (parameters is None) or type(self.model) == LinearModelWithBaseline:
            weights = pd.DataFrame(weights_vals,
                               columns=self.data.columns)
        else:
            weights = pd.DataFrame(weights_vals, index=self.parameters.index,
                               columns=self.data.columns)

        if fit_intercept:
            intercept = pd.Series(intercept_vals, index=self.data.columns,
                                  name='intercept')
            return weights, intercept

        return weights
