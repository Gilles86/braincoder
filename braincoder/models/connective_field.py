"""Connective-field (CF) models: target responses as a Gaussian-weighted pool
of source-region responses, with the Gaussian defined over cortical distance.

A connective field (Haak et al. 2013, *NeuroImage* 66:376-384) treats the
activity of a *source* region (typically V1, sampled on the cortical
surface) as the "stimulus" of a target voxel/vertex. The target's predicted
time course is a weighted sum of the source time courses, with weights given
by a Gaussian over geodesic distance on the cortical surface around a centre
vertex:

.. math::

    \\hat y_t(\\tau) = \\beta_0 + \\beta_1 \\sum_s w_s(c, \\sigma)\\, x_s(\\tau),
    \\qquad
    w_s(c, \\sigma) = \\exp\\left(-\\frac{d(c, s)^2}{2\\sigma^2}\\right)

where :math:`d(c, s)` is the geodesic distance between the CF centre
:math:`c` and source vertex :math:`s`. Distances between hemispheres are
``inf`` (zero weight). The fitted centre can be mapped to any property of the
source vertices, e.g. visual-field position from a retinotopic atlas.

The parameterisation follows ``prfmodel.models.cf.GaussianCFModel`` (popylar
project): parameters ``center_index`` (integer row of the distance matrix),
``sigma`` (in the units of the distance matrix, usually mm), ``baseline`` and
``amplitude``; ``normalize='density'`` reproduces prfmodel's
``1 / (2 pi sigma^2)`` weight scaling.

The centre is discrete and is found by grid search
(:class:`braincoder.optimize.ConnectiveFieldFitter`); ``sigma``, ``baseline``
and ``amplitude`` can afterwards be refined by gradient descent with the
ordinary :class:`braincoder.optimize.ParameterFitter` while ``center_index``
is held fixed.
"""
import math

import numpy as np
import pandas as pd
from keras import ops

from .base import EncodingModel
from ..utils import format_parameters

_NORMALIZE = (None, 'none', 'density', 'sum')


class GaussianConnectiveField(EncodingModel):
    """Gaussian connective field over a (geodesic) source distance matrix.

    Parameters
    ----------
    distance_matrix : array (n_sources, n_sources)
        Distances between source vertices, in mm. Use ``np.inf`` for pairs
        that cannot share a field (e.g. different hemispheres; see
        :func:`braincoder.utils.cortex.block_distance_matrix`). Compute it
        with :func:`braincoder.utils.cortex.geodesic_distance_matrix`.
    paradigm : array or DataFrame (n_samples, n_sources), optional
        Source-region time courses. This plays the role of the stimulus.
    data : DataFrame (n_samples, n_targets), optional
    parameters : DataFrame, optional
        Columns ``center_index, sigma, baseline, amplitude``.
    truncate : float or None
        Weights are set to zero beyond ``truncate * sigma``. ``None`` keeps
        the full Gaussian (prfmodel behaviour). Truncation makes the grid
        search sparse and is standard in CF work (e.g. 3 sigma).
    normalize : {'density', 'sum', None}
        ``'density'``: divide weights by ``2 pi sigma^2`` (prfmodel
        convention; amplitude is then per unit area of cortex). ``'sum'``:
        weights sum to one, so the pooled signal is a weighted mean of the
        sources and ``amplitude`` is a regression slope on it. ``None``:
        raw Gaussian. The choice does not affect the grid search (which uses
        correlation), only the scale of ``amplitude``.
    """

    parameter_labels = ['center_index', 'sigma', 'baseline', 'amplitude']

    def __init__(self, distance_matrix, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, truncate=3.0, normalize='density',
                 **kwargs):
        if normalize not in _NORMALIZE:
            raise ValueError(f'normalize must be one of {_NORMALIZE}, got {normalize!r}')
        D = np.asarray(distance_matrix, dtype=np.float32)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f'distance_matrix must be square, got shape {D.shape}')
        if paradigm is not None and np.shape(paradigm)[1] != D.shape[0]:
            raise ValueError(f'paradigm has {np.shape(paradigm)[1]} sources but the '
                             f'distance matrix has {D.shape[0]}')
        self.n_sources = D.shape[0]
        self.distance_matrix = D
        self.truncate = truncate
        self.normalize = None if normalize == 'none' else normalize
        self._D = ops.convert_to_tensor(D)
        self.transformations = ['identity', 'softplus', 'identity', 'identity']
        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, **kwargs)

    def _get_stimulus(self, n_dimensions=None, **kwargs):
        return super()._get_stimulus(n_dimensions=n_dimensions or self.n_sources)

    # ------------------------------------------------------------------ weights
    def _gaussian_weights(self, distances, sigma, normalize='model'):
        """Gaussian weights for a block of distance rows.

        distances : (..., n_sources), may contain ``inf``.
        sigma : broadcastable to ``distances[..., :1]``.
        normalize : 'model' uses ``self.normalize``; otherwise as in __init__.
        """
        normalize = self.normalize if normalize == 'model' else normalize
        finite = ops.isfinite(distances)
        # Never let inf enter exp(): its gradient w.r.t. sigma would be NaN.
        d = ops.where(finite, distances, ops.zeros_like(distances))
        w = ops.exp(-ops.square(d) / (2.0 * ops.square(sigma)))
        keep = finite
        if self.truncate is not None:
            keep = ops.logical_and(keep, d <= self.truncate * sigma)
        w = ops.where(keep, w, ops.zeros_like(w))
        if normalize == 'density':
            w = w / (2.0 * math.pi * ops.square(sigma))
        elif normalize == 'sum':
            w = w / ops.maximum(ops.sum(w, axis=-1, keepdims=True), 1e-12)
        return w

    def get_cf_weights(self, parameters=None):
        """Source weights of each target's connective field.

        Returns
        -------
        DataFrame (n_targets, n_sources)
        """
        parameters = self._get_parameters(parameters)
        c = ops.convert_to_tensor(parameters['center_index'].values.astype(np.int32))
        s = ops.convert_to_tensor(parameters['sigma'].values.astype(np.float32))[:, None]
        w = self._gaussian_weights(ops.take(self._D, c, axis=0), s)
        return pd.DataFrame(ops.convert_to_numpy(w), index=parameters.index)

    # ------------------------------------------------------------ prediction
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (batch, n_samples, n_sources)
        # parameters: (batch, n_targets, 4)
        centre = ops.cast(ops.round(parameters[..., 0]), 'int32')        # (b, v)
        sigma = parameters[..., 1:2]                                        # (b, v, 1)
        d = ops.take(self._D, centre, axis=0)                              # (b, v, s)
        w = self._gaussian_weights(d, sigma)
        pooled = ops.einsum('bts,bvs->btv', paradigm, w)
        return pooled * parameters[:, None, :, 3] + parameters[:, None, :, 2]

    def predict(self, paradigm=None, parameters=None, weights=None, batch_size=None):
        """Predicted target time courses.

        ``batch_size`` (targets per batch) bounds memory: each batch
        materialises an ``(n_targets, n_sources)`` weight matrix.
        """
        if weights is not None:
            raise ValueError('GaussianConnectiveField does not use weights')
        parameters = self._get_parameters(parameters)
        if batch_size is None or len(parameters) <= batch_size:
            return super().predict(paradigm, parameters)
        return pd.concat([super(GaussianConnectiveField, self).predict(
            paradigm, parameters.iloc[i:i + batch_size])
            for i in range(0, len(parameters), batch_size)], axis=1)

    # ------------------------------------------------------------ utilities
    def get_init_pars(self, data=None, paradigm=None, confounds=None):
        raise NotImplementedError(
            'The CF centre is discrete: initialise with '
            'ConnectiveFieldFitter.fit_grid() instead.')

    def source_properties(self, parameters=None, properties=None):
        """Map each target's CF centre to properties of that source vertex.

        Parameters
        ----------
        parameters : DataFrame with a ``center_index`` column
        properties : DataFrame (n_sources, k) or dict of arrays
            E.g. Benson x, y, eccentricity and polar angle of every source.
            Rows are in source order (positional, not by index label).

        Returns
        -------
        DataFrame (n_targets, k), indexed like ``parameters``.
        """
        parameters = self._get_parameters(parameters)
        properties = pd.DataFrame(properties)
        if len(properties) != self.n_sources:
            raise ValueError(f'properties has {len(properties)} rows, '
                             f'expected one per source ({self.n_sources})')
        c = parameters['center_index'].round().astype(int).values
        out = properties.iloc[c].copy()
        out.index = parameters.index
        return out

    def _get_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        if parameters is None:
            return None
        parameters = format_parameters(parameters, parameter_labels=self.parameter_labels)
        return parameters[self.parameter_labels]
