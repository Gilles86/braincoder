"""Gradient-based MAP over stimulus parameters (see :mod:`braincoder.parametric`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import keras
from keras import ops

from ..utils.backend import compute_gradients, mvn_log_prob, mvt_log_prob, safe_cholesky


class ParametricStimulusFitter(object):
    """MAP of stimulus parameters φ given data, by gradient descent through ``stimulus.to_image``.

    Two modes, set by the model:

    * a model without HRF (single responses or betas): every row of ``data`` is
      one independent stimulus, φ_t is fitted per row;
    * a ``*WithHRF`` model and a BOLD time series: the images of all frames go
      through the HRF together, so the fit is over the whole sequence φ_1..φ_n.
      An optional Gaussian random-walk prior (``smoothness``) couples
      neighbouring frames.

    The stimulus must be differentiable (e.g. ``BarStimulus(edge='soft')``).
    The objective is ``-sum_t log p(y_t | f(g(φ_t))) / T - log prior``; the
    likelihood is the noise model's (``LowRankNoise`` or dense ``omega``/``dof``).

    Parameters
    ----------
    model : EncodingModel
    stimulus : ParametricStimulus
    data : (n, n_voxels) DataFrame or array
    parameters, weights : encoding parameters (as for ``predict``)
    noise_model : LowRankNoise, optional
    omega, dof : dense noise covariance and Student-t dof, optional
    temperature : float, optional (default: the noise model's, else 1)
    """

    def __init__(self, model, stimulus, data, parameters=None, weights=None, noise_model=None, omega=None, dof=None,
                 temperature=None):
        self.model, self.stimulus = model, stimulus
        self.index = data.index if hasattr(data, 'index') else pd.RangeIndex(len(data))
        self.data = np.asarray(data.values if hasattr(data, 'values') else data, dtype=np.float32)
        pars = model._get_parameters(parameters)
        self.parameters = np.asarray(pars.values if hasattr(pars, 'values') else pars, dtype=np.float32)
        self.weights = None if weights is None else np.asarray(weights.values if hasattr(weights, 'values') else weights, dtype=np.float32)
        if (noise_model is None) == (omega is None):
            raise ValueError('pass exactly one of noise_model or omega')
        self.noise_model, self.dof = noise_model, dof
        self.omega_chol = None if omega is None else safe_cholesky(ops.convert_to_tensor(np.asarray(omega, np.float32)))
        if temperature is None:
            temperature = getattr(noise_model, 'temperature', 1.) if noise_model is not None else 1.
        self.temperature = float(temperature)
        self.sequence = hasattr(model, 'hrf_model')

    def _log_likelihood(self, images):
        pred = self.model._predict(images[None], self.parameters[None], None if self.weights is None else self.weights[None])[0]
        resid = ops.convert_to_tensor(self.data) - pred
        if self.noise_model is not None:
            return self.noise_model.log_prob_backend(resid)
        if self.dof is None:
            return mvn_log_prob(resid, self.omega_chol)
        return mvt_log_prob(resid, self.omega_chol, ops.convert_to_tensor(float(self.dof)))

    def log_posterior(self, phi, smoothness=None, use_width_prior=False):
        """Objective (log likelihood / T + log prior) at ``phi`` (DataFrame or array), as a float."""
        u = ops.convert_to_tensor(self.stimulus.to_unconstrained(phi))
        return float(ops.convert_to_numpy(-self._loss(u, smoothness, use_width_prior)))

    def _loss(self, u, smoothness, use_width_prior=False):
        phi = self.stimulus.from_unconstrained(u)
        ll = ops.sum(self._log_likelihood(self.stimulus.to_image(phi))) / self.temperature
        pen = 0.
        if use_width_prior:
            pen = pen - self.stimulus.log_prior_unconstrained(u)
        if smoothness:
            for label, sd in smoothness.items():
                i = self.stimulus.parameter_labels.index(label)
                d = phi[1:, i] - phi[:-1, i]
                if label == 'orientation':                     # wrap to (-90, 90]: the bar's 180° symmetry
                    r = d * (np.pi / 90.)
                    d = ops.arctan2(ops.sin(r), ops.cos(r)) * (90. / np.pi)
                pen = pen + .5 * ops.sum((d / sd) ** 2)
        return -ll + pen

    def fit(self, init_phi, max_n_iterations=500, learning_rate=.05, fixed=None, smoothness=None, rtol=1e-7,
            min_n_iterations=50, lag=50, decay=True, progressbar=False, use_width_prior=False):
        """Gradient descent (Adam) on φ in the stimulus' unconstrained space.

        Parameters
        ----------
        init_phi : DataFrame (columns ``stimulus.parameter_labels``) or array, one row per frame
            Start, e.g. the grid MAP of :func:`~braincoder.parametric.get_parametric_stimulus_pdf`.
        fixed : list of parameter labels, optional
            Kept at their start values (clamped), e.g. ``['orientation', 'width']``.
        smoothness : dict label -> SD per frame, optional
            Gaussian random-walk prior over frames (sequence mode); orientation
            differences are wrapped to ±90°.
        decay : bool
            Cosine decay of the learning rate over ``max_n_iterations`` (default True).
        use_width_prior : bool
            Add the stimulus' ``width_prior`` (a :class:`~braincoder.parametric.LogWidthPrior`, in its unconstrained
            coordinate) to the objective: a MAP with a proper bounded prior on width instead of a flat one.

        Returns
        -------
        DataFrame of fitted φ (canonicalized), index as ``data``. The loss trace is in ``self.costs``.
        """
        if isinstance(init_phi, pd.DataFrame):
            init_phi = init_phi[self.stimulus.parameter_labels].values
        init_phi = np.asarray(init_phi, np.float64)
        if len(init_phi) != len(self.data):
            raise ValueError('init_phi needs one row per row of data')
        u0 = self.stimulus.to_unconstrained(init_phi)
        labels = self.stimulus.parameter_labels
        fixed = [] if fixed is None else list(fixed)
        free = [i for i, l in enumerate(labels) if l not in fixed]
        if not free:
            raise ValueError('nothing to fit')
        var = keras.Variable(u0[:, free], dtype='float32', name='phi')
        const = ops.convert_to_tensor(u0)
        n_par = len(labels)

        def build():
            cols, j = [], 0
            for i in range(n_par):
                if i in free:
                    cols.append(var[:, j:j + 1])
                    j += 1
                else:
                    cols.append(const[:, i:i + 1])
            return ops.concatenate(cols, axis=1)

        # Cosine decay to 1% of the rate: a fixed Adam step keeps oscillating in a sharp likelihood.
        schedule = keras.optimizers.schedules.CosineDecay(learning_rate, max_n_iterations, alpha=.01) if decay else learning_rate
        opt = keras.optimizers.Adam(learning_rate=schedule)
        costs = []
        it = range(max_n_iterations)
        if progressbar:
            from tqdm.auto import tqdm
            it = tqdm(it)
        for step in it:
            loss, grads = compute_gradients(lambda: self._loss(build(), smoothness, use_width_prior), [var])
            opt.apply_gradients(zip(grads, [var]))
            costs.append(float(ops.convert_to_numpy(loss)))
            if step > min_n_iterations and abs(costs[-1 - lag] - costs[-1]) <= rtol * abs(costs[-1 - lag]):
                break
        self.costs = np.array(costs)
        phi = np.asarray(ops.convert_to_numpy(self.stimulus.from_unconstrained(build())), dtype=np.float64)
        out = pd.DataFrame(phi, columns=labels, index=self.index)
        return self.stimulus.canonicalize(out)
