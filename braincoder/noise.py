"""Low-rank + diagonal noise models for decoding, with cross-validated
rank and likelihood temperature.

Quick start
-----------
>>> from braincoder.noise import LowRankNoise
>>> noise = LowRankNoise.fit(model, train_data, train_paradigm, parameters,
...                          stimulus_range=grid, groups=train_runs)
>>> pdf = model.get_stimulus_pdf(test_data, stimulus_range=grid,
...                              parameters=parameters, noise_model=noise)
>>> print(noise.summary())

The noise covariance is

.. math::

    \\Omega = D\\,(F F^\\top + \\Psi)\\,D,

with :math:`D = \\mathrm{diag}(s)` the per-voxel residual scale, :math:`F`
(``n_voxels × k``) the first ``k`` principal factors of the residual
*correlation* matrix and :math:`\\Psi` the diagonal uniquenesses (floored at
``psi_floor``). The likelihood is a multivariate Gaussian or Student-t with
scale :math:`\\Omega`. It is never formed as a dense ``V × V`` matrix: the
Mahalanobis distance and log-determinant use the Woodbury identity and the
matrix-determinant lemma, so cost is ``O(V k)`` per residual vector.

Decoding posteriors are *tempered*: :math:`p(s \\mid y) \\propto
p(y \\mid s)^{1/T}`. ``k`` and ``T`` are chosen jointly by **nested,
grouped cross-validation on the decoding objective** — the held-out log
posterior probability of the true stimulus — not by how well the
covariance fits the residuals. (On real 7T single-trial betas the
covariance likelihood keeps asking for more factors while decoding gets
worse; see the user guide.)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

__all__ = ['LowRankNoise', 'fit_lowrank_omega', 'DEFAULT_K_GRID',
           'DEFAULT_TEMPERATURE_GRID']

logger = logging.getLogger(__name__)

#: Default candidate ranks (capped by the number of inner-training trials
#: and voxels at fit time).
DEFAULT_K_GRID = (0, 1, 2, 5, 10, 20, 40, 80)

#: Default candidate temperatures: 2^-2 … 2^12 in steps of 2^(1/4), plus
#: ``inf`` (= flat posterior; chosen when the likelihood carries no usable
#: calibrated information).
DEFAULT_TEMPERATURE_GRID = tuple(np.r_[2.0 ** np.arange(-2, 12.001, 0.25), np.inf])

_DOF_BOUNDS = (2.05, 1e4)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_2d_array(x):
    if hasattr(x, 'values'):
        x = x.values
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _standardise(R):
    mu = R.mean(0)
    sd = R.std(0)
    good = sd > 0
    fill = np.median(sd[good]) if good.any() else 1.0
    sd = np.where(good, sd, fill)
    return (R - mu) / sd, sd


def _match_to_grid(stimuli, grid, name='paradigm'):
    """Index of the nearest grid point for every row of ``stimuli``."""
    stimuli = _as_2d_array(stimuli)
    grid = _as_2d_array(grid)
    if stimuli.shape[1] != grid.shape[1]:
        raise ValueError(f'{name} has {stimuli.shape[1]} stimulus dimensions, '
                         f'stimulus_range has {grid.shape[1]}')
    d2 = ((stimuli[:, None, :] - grid[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    if len(grid) > 1:
        gd = ((grid[:, None, :] - grid[None, :, :]) ** 2).sum(-1)
        gd[np.diag_indices_from(gd)] = np.inf
        half_spacing2 = (0.5 ** 2) * gd.min(1).max()
        if d2[np.arange(len(idx)), idx].max() > half_spacing2 * 1.0001:
            warnings.warn(f'Some {name} stimuli lie more than half a grid step '
                          'from the nearest point of stimulus_range; their '
                          'true stimulus is approximated by the nearest grid point.',
                          RuntimeWarning, stacklevel=3)
    return idx


def _mvt_logpdf(d2, logdet, V, dof):
    if dof is None or not np.isfinite(dof):
        return -0.5 * (V * np.log(2 * np.pi) + logdet + d2)
    return (gammaln((dof + V) / 2.) - gammaln(dof / 2.) - 0.5 * V * np.log(dof * np.pi)
            - 0.5 * logdet - 0.5 * (dof + V) * np.log1p(d2 / dof))


def _fit_scale_dof(d2, logdet, V, dof='fit'):
    """ML (c, ν) for residuals with Mahalanobis d2 under unit-scale Ω.

    Ω_fitted = c·Ω. ``dof``: 'fit' (ML, 1-D bounded search on log ν with c
    profiled out), ``None`` (Gaussian, closed-form c) or a fixed number.
    Returns (c, dof, total_loglik).
    """
    d2 = np.asarray(d2, dtype=np.float64)
    n = len(d2)

    if dof is None:
        c = max(d2.mean() / V, 1e-300)
        return c, None, float(_mvt_logpdf(d2 / c, logdet + V * np.log(c), V, None).sum())

    def best_c(nu):
        # EM fixed point for the scale of a multivariate t, started at the
        # Gaussian estimate; converges monotonically.
        c = max(d2.mean() / V, 1e-300)
        for _ in range(200):
            w = (nu + V) / (nu + d2 / c)
            c_new = (w * d2).sum() / (n * V)
            if abs(c_new - c) <= 1e-10 * c:
                c = c_new
                break
            c = c_new
        ll = _mvt_logpdf(d2 / c, logdet + V * np.log(c), V, nu).sum()
        return c, ll

    if dof == 'fit':
        res = minimize_scalar(lambda lnu: -best_c(np.exp(lnu))[1],
                              bounds=np.log(_DOF_BOUNDS), method='bounded',
                              options=dict(xatol=1e-3))
        nu = float(np.exp(res.x))
        # also compare against the boundaries (the profile can be monotone)
        cands = [nu, _DOF_BOUNDS[0], _DOF_BOUNDS[1]]
        lls = [best_c(c)[1] for c in cands]
        nu = cands[int(np.argmax(lls))]
    else:
        nu = float(dof)
    c, ll = best_c(nu)
    return float(c), nu, float(ll)


# ---------------------------------------------------------------------------
# the noise model
# ---------------------------------------------------------------------------

class LowRankNoise(object):
    """Low-rank + diagonal (Gaussian or Student-t) noise model for decoding.

    Ω = D (F Fᵀ + Ψ) D, stored in factored form. Construct with
    :meth:`fit` (recommended: chooses ``k`` and the temperature ``T`` by
    nested grouped cross-validation on the decoding objective) or
    :meth:`from_residuals` (covariance only, fixed ``k``).

    Attributes
    ----------
    k : int
        Number of factors.
    temperature : float
        Likelihood temperature applied by ``get_stimulus_pdf`` (``inf`` =
        flat posterior).
    dof : float or None
        Student-t degrees of freedom (``None`` = Gaussian).
    scale : ndarray (V,)
        Per-voxel scale ``s`` (includes the ML scale factor).
    factors : ndarray (V, k)
        ``F``, loadings on the correlation scale.
    psi : ndarray (V,)
        Diagonal uniquenesses on the correlation scale.
    voxels : pandas.Index or None
        Voxel labels (columns of the training data), used to align test data.
    cv_scores_ : pandas.DataFrame or None
        Cross-validated decoding score for every (k, T): mean held-out log
        posterior probability of the true stimulus per trial, **minus** that
        of a flat posterior (nats; > 0 = better than uniform). Rows = k,
        columns = T.
    """

    def __init__(self, scale, factors, psi, dof=None, temperature=1.0,
                 voxels=None):
        self.scale = np.asarray(scale, dtype=np.float64)
        V = len(self.scale)
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim == 1:
            factors = factors[:, None]
        if factors.size == 0:
            factors = np.zeros((V, 0))
        self.factors = factors
        self.psi = np.asarray(psi, dtype=np.float64)
        if self.factors.shape[0] != V or self.psi.shape != (V,):
            raise ValueError('scale, factors and psi must have matching n_voxels')
        if np.any(self.psi <= 0) or np.any(self.scale <= 0):
            raise ValueError('psi and scale must be strictly positive')
        self.dof = None if dof is None or not np.isfinite(dof) else float(dof)
        self.temperature = float(temperature)
        self.voxels = None if voxels is None else pd.Index(voxels)
        self.cv_scores_ = None
        self.cv_info_ = {}
        self._precompute()

    # -- factored algebra ----------------------------------------------------

    def _precompute(self):
        F, psi = self.factors, self.psi
        k = F.shape[1]
        FtPinv = (F / psi[:, None]).T                          # k x V
        M = np.eye(k) + FtPinv @ F
        Lm = np.linalg.cholesky(M)
        # A = Lm^{-1} F^T Psi^{-1}; Ω_C^{-1} = Ψ^{-1} - Aᵀ A
        self._A = np.linalg.solve(Lm, FtPinv) if k > 0 else np.zeros((0, len(psi)))
        self._sqrt_inv_psi = 1.0 / np.sqrt(psi)
        self.logdet = float(2 * np.log(self.scale).sum() + np.log(psi).sum()
                            + 2 * np.log(np.diag(Lm)).sum()) if k > 0 else \
            float(2 * np.log(self.scale).sum() + np.log(psi).sum())
        self._tf_cache = None

    @property
    def k(self):
        return self.factors.shape[1]

    @property
    def n_voxels(self):
        return len(self.scale)

    def _components(self, X):
        """(G, H) such that <a, b>_{Ω⁻¹} = G_a·G_b − H_a·H_b."""
        Z = X / self.scale
        return Z * self._sqrt_inv_psi, Z @ self._A.T

    def mahalanobis(self, residuals):
        """Squared Mahalanobis distance rᵀ Ω⁻¹ r of each row (Woodbury)."""
        R = self._align(residuals)
        G, H = self._components(R)
        return np.clip((G ** 2).sum(1) - (H ** 2).sum(1), 0, None)

    def log_prob(self, residuals):
        """Log density of each row of ``residuals`` (numpy, float64)."""
        d2 = self.mahalanobis(residuals)
        return _mvt_logpdf(d2, self.logdet, self.n_voxels, self.dof)

    def grid_log_likelihood(self, data, predictions):
        """Log-likelihood table ``(n_trials, n_stimuli)``.

        ``data``: (n_trials, V); ``predictions``: (n_stimuli, V), the model
        prediction for every candidate stimulus. Computed as
        d²(t, s) = q(y_t) + q(μ_s) − 2⟨y_t, μ_s⟩ with all terms in factored
        form — one (n_trials × V) · (V × n_stimuli) product, no dense Ω and
        no (n_stimuli × n_trials × V) residual tensor. float64 numpy.
        """
        Y = self._align(data)
        P = _as_2d_array(predictions)
        Gy, Hy = self._components(Y)
        Gp, Hp = self._components(P)
        d2 = ((Gy ** 2).sum(1) - (Hy ** 2).sum(1))[:, None] \
            + ((Gp ** 2).sum(1) - (Hp ** 2).sum(1))[None, :] \
            - 2 * (Gy @ Gp.T - Hy @ Hp.T)
        d2 = np.clip(d2, 0, None)
        return _mvt_logpdf(d2, self.logdet, self.n_voxels, self.dof)

    def log_prob_backend(self, residuals):
        """Log density of each row, in Keras ops (float32, differentiable).

        Same factored computation as :meth:`log_prob`, usable inside a
        TensorFlow / JAX / PyTorch graph (e.g. gradient-based stimulus
        fitting). ``residuals``: tensor (n, V).
        """
        from keras import ops
        from .utils.backend import lgamma
        if self._tf_cache is None:
            self._tf_cache = dict(
                inv_s=ops.convert_to_tensor((1. / self.scale).astype(np.float32)),
                sip=ops.convert_to_tensor(self._sqrt_inv_psi.astype(np.float32)),
                At=ops.convert_to_tensor(self._A.T.astype(np.float32)) if self.k > 0 else None,
            )
        c = self._tf_cache
        z = ops.cast(residuals, 'float32') * c['inv_s']
        q = ops.sum((z * c['sip']) ** 2, axis=1)
        if c['At'] is not None:
            q = q - ops.sum(ops.matmul(z, c['At']) ** 2, axis=1)
        q = ops.maximum(q, 0.)
        V = float(self.n_voxels)
        if self.dof is None:
            return -0.5 * (V * np.log(2 * np.pi) + self.logdet + q)
        nu = self.dof
        const = float(gammaln((nu + V) / 2.) - gammaln(nu / 2.)
                      - 0.5 * V * np.log(nu * np.pi) - 0.5 * self.logdet)
        return const - 0.5 * (nu + V) * ops.log1p(q / nu)

    def to_dense(self):
        """Dense ``(omega, dof)``, drop-in for ``ResidualFitter.fit`` output.

        For code paths that take ``omega=`` / ``dof=``. Note the temperature
        is *not* part of (omega, dof); pass ``temperature=noise.temperature``
        to ``get_stimulus_pdf`` as well.
        """
        C = self.factors @ self.factors.T
        C[np.diag_indices_from(C)] += self.psi
        omega = C * self.scale[:, None] * self.scale[None, :]
        return omega.astype(np.float32), self.dof

    def _align(self, X):
        if self.voxels is not None and isinstance(X, pd.DataFrame):
            if not X.columns.equals(self.voxels):
                missing = self.voxels.difference(X.columns)
                if len(missing):
                    raise ValueError(f'{len(missing)} voxels of the noise model are '
                                     'missing from the data')
                X = X.loc[:, self.voxels]
        X = _as_2d_array(X)
        if X.shape[1] != self.n_voxels:
            raise ValueError(f'data has {X.shape[1]} voxels, noise model {self.n_voxels}')
        return X

    # -- construction --------------------------------------------------------

    @staticmethod
    def _svd(R):
        Z, sd = _standardise(R)
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        return dict(sd=sd, s=s, Vt=Vt, n=len(R))

    @staticmethod
    def _structure(svd, k, psi_floor):
        """Unit-scale (sd, F, psi) for rank k from a precomputed SVD."""
        lam = svd['s'][:k] ** 2 / svd['n']
        F = (svd['Vt'][:k] * np.sqrt(lam)[:, None]).T          # V x k
        common = (F ** 2).sum(1)
        psi = np.clip(1.0 - common, psi_floor, None)
        return svd['sd'], F, psi

    @classmethod
    def from_residuals(cls, residuals, k=40, dof='fit', psi_floor=0.02,
                       temperature=1.0, _svd=None):
        """Fit Ω (and ν) to residuals for a fixed rank ``k``.

        Principal-factor model of the residual correlation (one closed-form
        step, no iteration): F = first k eigenvectors × √eigenvalues, Ψ =
        1 − diag(F Fᵀ), floored at ``psi_floor``; then an overall scale c and
        ν are fitted by maximum likelihood (Student-t; ``dof=None`` for
        Gaussian, a number to fix ν).

        Parameters
        ----------
        residuals : DataFrame or array (n_trials, V)
        k : int
            Number of factors (capped at ``min(n_trials, V) − 1``).
        """
        voxels = residuals.columns if isinstance(residuals, pd.DataFrame) else None
        R = _as_2d_array(residuals)
        n, V = R.shape
        k = int(min(k, max(min(n, V) - 1, 0)))
        svd = cls._svd(R) if _svd is None else _svd
        sd, F, psi = cls._structure(svd, k, psi_floor)
        unit = cls(sd, F, psi, dof=None, voxels=voxels)
        d2 = unit.mahalanobis(R)
        c, nu, ll = _fit_scale_dof(d2, unit.logdet, V, dof=dof)
        out = cls(sd * np.sqrt(c), F, psi, dof=nu, temperature=temperature,
                  voxels=voxels)
        out.train_loglik_ = ll
        out.psi_floor = psi_floor
        return out

    @classmethod
    def fit(cls, model, data, paradigm, parameters=None, stimulus_range=None,
            groups=None, weights=None, k='cv', temperature='cv', dof='fit',
            k_grid=DEFAULT_K_GRID, temperature_grid=DEFAULT_TEMPERATURE_GRID,
            inner_parameters=None, psi_floor=0.02, n_folds=5,
            progressbar=False):
        """Fit a low-rank noise model; choose ``k`` and ``T`` by nested CV.

        Parameters
        ----------
        model : EncodingModel
        data : DataFrame (n_trials, V)
            **Training** data only (the test data must never be passed here).
        paradigm : DataFrame or array (n_trials, n_stim_dims)
            True stimulus of every training trial.
        parameters : DataFrame, optional
            Encoding parameters (default ``model.parameters``).
        stimulus_range : array (n_stimuli,) or (n_stimuli, n_dims)
            The decoding grid — use the same grid as for
            ``get_stimulus_pdf``. Required when ``k`` or ``temperature`` is
            chosen by CV. True training stimuli are matched to the nearest
            grid point.
        groups : array (n_trials,), optional
            Group label per training trial (e.g. run). Inner CV leaves one
            group out at a time. Strongly recommended: trials within a run
            share noise. Without it, ``n_folds`` contiguous blocks are used.
        k : 'cv' or int
        temperature : 'cv' or float (``np.inf`` = flat posterior)
        dof : 'fit', None or float
            Student-t ν by ML ('fit'), Gaussian (``None``) or fixed.
        k_grid, temperature_grid : sequences
            Candidates for CV. ``k`` values ≥ the number of inner-training
            trials or voxels are dropped.
        inner_parameters : None, dict or callable
            Encoding parameters used inside inner fold ``g``.

            * ``None`` (default): ``parameters`` are **held fixed**. The
              inner held-out trials were then part of the data the
              encoding model was fitted on, so their predictions are
              slightly optimistic and the chosen ``T`` is, if anything,
              too small (posteriors a bit too sharp).
            * ``dict`` {group: parameters}: precomputed parameters that did
              not see group ``g`` (e.g. existing leave-one-run-out fits).
            * callable ``f(train_data, train_paradigm, group) -> parameters``:
              refit the encoding model on the inner-training trials.

            For weight-based models (a fixed basis set whose per-voxel
            ``weights`` are what is fitted, e.g. via ``WeightFitter``), the
            dict value / callable return may instead be a
            ``(parameters, weights)`` tuple: the inner fold then uses those
            weights rather than the outer ``weights``. Otherwise the outer
            ``weights`` are used in every inner fold, so a basis-set model
            is effectively decoded with fixed (optimistic) inner parameters.

            In fold ``g``, Ω is fitted on the residuals of the inner-training
            trials under these parameters, and the held-out trials are
            decoded with them.
        psi_floor : float
            Floor on the diagonal uniquenesses (correlation scale).

        Returns
        -------
        LowRankNoise
            Fitted on *all* training trials with the chosen ``k``; carries
            the chosen ``temperature``, ``cv_scores_`` and ``cv_info_``.

        Notes
        -----
        The CV objective is the summed held-out log posterior probability of
        the true stimulus on the grid, ``log p_T(s* | y)``, under a flat
        prior over the grid. This is the decoding objective; it is *not* the
        covariance likelihood (which prefers ever larger ``k`` on fMRI data
        while decoding gets worse). For every inner fold and every ``k``,
        Ω(k) and ν are refitted on the inner-training residuals; all ``T``
        are evaluated on the same likelihood table at no extra cost.
        """
        if parameters is None:
            parameters = model.parameters
        voxels = data.columns if isinstance(data, pd.DataFrame) else None
        Y = _as_2d_array(data)
        n, V = Y.shape

        def residuals_for(pars, idx, w=weights):
            pred = model.predict(paradigm=_subset(paradigm, idx),
                                 parameters=pars, weights=w)
            return Y[idx] - _as_2d_array(pred)

        all_idx = np.arange(n)
        R_all = residuals_for(parameters, all_idx)

        need_cv = (isinstance(k, str) and k == 'cv') or \
                  (isinstance(temperature, str) and temperature == 'cv')

        if not need_cv:
            out = cls.from_residuals(R_all, k=int(k), dof=dof, psi_floor=psi_floor,
                                     temperature=float(temperature))
            out.voxels = None if voxels is None else pd.Index(voxels)
            return out

        if stimulus_range is None:
            raise ValueError('stimulus_range is required to choose k / temperature '
                             'by cross-validation')

        grid = _as_2d_array(stimulus_range)
        true_idx = _match_to_grid(paradigm, grid)
        S = len(grid)

        if groups is None:
            warnings.warn('No `groups` given: inner CV uses %d contiguous blocks of '
                          'trials. Pass run labels if trials within a run share '
                          'noise.' % n_folds, RuntimeWarning, stacklevel=2)
            groups = np.repeat(np.arange(n_folds), int(np.ceil(n / n_folds)))[:n]
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError('groups must have one entry per training trial')
        ugroups = pd.unique(groups)
        if len(ugroups) < 2:
            raise ValueError('need at least two groups for cross-validation')

        n_tr_min = min((groups != g).sum() for g in ugroups)
        if isinstance(k, str):
            ks = sorted({int(kk) for kk in k_grid if kk < min(n_tr_min, V)})
        else:
            ks = [int(k)]
        if isinstance(temperature, str):
            Ts = np.asarray(temperature_grid, dtype=float)
        else:
            Ts = np.asarray([float(temperature)])

        finite_T = Ts[np.isfinite(Ts)]
        scores = np.zeros((len(ks), len(Ts)))
        nus = {kk: [] for kk in ks}

        iterator = ugroups
        if progressbar:
            from tqdm.auto import tqdm
            iterator = tqdm(ugroups, desc='inner CV')

        for g in iterator:
            te = np.where(groups == g)[0]
            tr = np.where(groups != g)[0]
            w_g = weights
            if inner_parameters is None:
                pars_g = parameters
                R_tr = R_all[tr]
            else:
                if callable(inner_parameters):
                    pars_g = inner_parameters(_subset(data, tr), _subset(paradigm, tr), g)
                else:
                    pars_g = inner_parameters[g]
                if isinstance(pars_g, tuple):
                    pars_g, w_g = pars_g
                R_tr = residuals_for(pars_g, tr, w_g)

            P = _grid_predictions(model, grid, pars_g, w_g)        # S x V
            svd = cls._svd(R_tr)
            for ki, kk in enumerate(ks):
                nm = cls.from_residuals(R_tr, k=kk, dof=dof, psi_floor=psi_floor,
                                        _svd=svd)
                nus[kk].append(nm.dof)
                ll = nm.grid_log_likelihood(Y[te], P)               # n_te x S
                ll_true = ll[np.arange(len(te)), true_idx[te]]
                for ti, T in enumerate(Ts):
                    if np.isinf(T):
                        scores[ki, ti] += -len(te) * np.log(S)
                    else:
                        a = ll / T
                        scores[ki, ti] += (ll_true / T - logsumexp(a, axis=1)).sum()

        # mean per trial, relative to the flat posterior
        rel = scores / n + np.log(S)
        cv = pd.DataFrame(rel, index=pd.Index(ks, name='k'),
                          columns=pd.Index(Ts, name='temperature'))
        ki, ti = np.unravel_index(np.nanargmax(rel), rel.shape)
        k_best, T_best = ks[ki], float(Ts[ti])

        out = cls.from_residuals(R_all, k=k_best, dof=dof, psi_floor=psi_floor,
                                 temperature=T_best)
        out.voxels = None if voxels is None else pd.Index(voxels)
        out.cv_scores_ = cv
        out.cv_info_ = dict(n_trials=n, n_voxels=V, n_groups=len(ugroups),
                            n_stimuli=S, inner_parameters=(
                                'fixed' if inner_parameters is None else
                                'callable' if callable(inner_parameters) else 'dict'),
                            inner_dof={kk: nus[kk] for kk in ks})
        if len(finite_T) > 1 and T_best == finite_T.max():
            warnings.warn('Chosen temperature is the largest finite value in the '
                          'grid; consider extending temperature_grid.',
                          RuntimeWarning, stacklevel=2)
        if isinstance(k, str) and k_best == max(ks) and len(ks) > 1:
            warnings.warn('Chosen k is the largest value in k_grid; consider '
                          'extending it.', RuntimeWarning, stacklevel=2)
        return out

    @classmethod
    def fit_select_voxels(cls, model, data, paradigm, parameters=None,
                          stimulus_range=None, groups=None, weights=None,
                          n_voxels_grid=(25, 50, 100, 200, 400, 800, None),
                          inner_parameters=None, k_grid=DEFAULT_K_GRID,
                          temperature_grid=DEFAULT_TEMPERATURE_GRID,
                          max_k_fraction=0.25, dof='fit', psi_floor=0.02):
        """Choose the number of voxels N jointly with ``k`` and ``T`` by nested CV.

        Voxels are ranked by their in-sample R² under the encoding model, on
        training trials only: inside inner fold ``g`` by the R² of that fold's
        inner parameters on its inner-training trials, and for the returned
        model by the R² of ``parameters`` on all training trials. For each
        candidate N the top-N voxels of the fold's own ranking are taken and
        Ω(k) is refitted on them (the factors of a voxel subset are not a
        subset of the factors of a larger set), then scored on the held-out
        group exactly as in :meth:`fit`. The objective is the same held-out
        log posterior of the true grid stimulus.

        Parameters
        ----------
        data : DataFrame (n_trials, V)
            Training data of ALL candidate voxels (e.g. a whole ROI). Must be
            a DataFrame: the selected voxels are returned by label.
        n_voxels_grid : sequence of int or None
            Candidate N; ``None`` = all voxels. Values above V are capped at V.
        inner_parameters : None, dict or callable
            As in :meth:`fit`, but for ALL voxels (``(parameters, weights)``
            tuples allowed). ``None`` holds ``parameters`` fixed, which also
            fixes the ranking across inner folds (optimistic).
        max_k_fraction : float
            Only ranks ``k <= max_k_fraction * N`` are considered for N.

        Returns
        -------
        LowRankNoise
            Fitted on the selected voxels of all training trials. Extra
            attributes: ``voxels`` (selected labels, in ranking order),
            ``n_voxels_selected_``, ``voxel_ranking_`` (all labels, best first)
            and ``cv_scores_`` with a (n_voxels, k) row MultiIndex.
            Decode with the selected voxels only, i.e. subset ``parameters``
            (or ``weights``) and the test data to ``noise.voxels``.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('fit_select_voxels needs data as a DataFrame (voxel labels)')
        if stimulus_range is None:
            raise ValueError('stimulus_range is required')
        if parameters is None:
            parameters = model.parameters
        labels = pd.Index(data.columns)
        Y = _as_2d_array(data)
        n, V = Y.shape
        grid = _as_2d_array(stimulus_range)
        true_idx = _match_to_grid(paradigm, grid)
        S = len(grid)
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError('groups must have one entry per training trial')
        ugroups = pd.unique(groups)
        if len(ugroups) < 2:
            raise ValueError('need at least two groups for cross-validation')
        n_tr_min = min((groups != g).sum() for g in ugroups)
        Ns = sorted({V if N is None else int(min(N, V)) for N in n_voxels_grid})
        Ns = [N for N in Ns if N >= 2]
        Ts = np.asarray(temperature_grid, dtype=float)

        def ks_for(N):
            ks = sorted({int(kk) for kk in k_grid
                         if kk < min(n_tr_min, N) and kk <= max_k_fraction * N})
            return ks or [0]

        def predict(pars, idx, w):
            return _as_2d_array(model.predict(paradigm=_subset(paradigm, idx),
                                              parameters=pars, weights=w))

        def r2_order(Ysub, pred):
            ss_res = ((Ysub - pred) ** 2).sum(0)
            ss_tot = ((Ysub - Ysub.mean(0)) ** 2).sum(0)
            r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.inf)
            return np.argsort(-r2, kind='stable')

        scores = {(N, kk): np.zeros(len(Ts)) for N in Ns for kk in ks_for(N)}
        for g in ugroups:
            te = np.where(groups == g)[0]
            tr = np.where(groups != g)[0]
            pars_g, w_g = parameters, weights
            if inner_parameters is not None:
                pars_g = (inner_parameters(_subset(data, tr), _subset(paradigm, tr), g)
                          if callable(inner_parameters) else inner_parameters[g])
                if isinstance(pars_g, tuple):
                    pars_g, w_g = pars_g
            pred_tr = predict(pars_g, tr, w_g)
            R_tr_full = Y[tr] - pred_tr
            order = r2_order(Y[tr], pred_tr)
            P_full = _grid_predictions(model, grid, pars_g, w_g)       # S x V
            for N in Ns:
                cols = order[:N]
                R_tr = R_tr_full[:, cols]
                svd = cls._svd(R_tr)
                for kk in ks_for(N):
                    nm = cls.from_residuals(R_tr, k=kk, dof=dof, psi_floor=psi_floor,
                                            _svd=svd)
                    ll = nm.grid_log_likelihood(Y[te][:, cols], P_full[:, cols])
                    ll_true = ll[np.arange(len(te)), true_idx[te]]
                    sc = scores[(N, kk)]
                    for ti, T in enumerate(Ts):
                        if np.isinf(T):
                            sc[ti] += -len(te) * np.log(S)
                        else:
                            sc[ti] += (ll_true / T - logsumexp(ll / T, axis=1)).sum()

        keys = list(scores)
        rel = np.array([scores[key] for key in keys]) / n + np.log(S)
        cv = pd.DataFrame(rel, index=pd.MultiIndex.from_tuples(keys, names=['n_voxels', 'k']),
                          columns=pd.Index(Ts, name='temperature'))
        ri, ti = np.unravel_index(np.nanargmax(rel), rel.shape)
        N_best, k_best = keys[ri]
        T_best = float(Ts[ti])

        pred_all = predict(parameters, np.arange(n), weights)
        order = r2_order(Y, pred_all)
        cols = order[:N_best]
        out = cls.from_residuals((Y - pred_all)[:, cols], k=k_best, dof=dof,
                                 psi_floor=psi_floor, temperature=T_best)
        out.voxels = labels[cols]
        out.n_voxels_selected_ = int(N_best)
        out.voxel_ranking_ = labels[order]
        out.cv_scores_ = cv
        out.cv_info_ = dict(n_trials=n, n_voxels=V, n_groups=len(ugroups), n_stimuli=S,
                            n_voxels_grid=Ns, inner_parameters=(
                                'fixed' if inner_parameters is None else
                                'callable' if callable(inner_parameters) else 'dict'))
        if N_best == max(Ns) and len(Ns) > 1 and max(Ns) < V:
            warnings.warn('Chosen n_voxels is the largest candidate below V; consider '
                          'extending n_voxels_grid.', RuntimeWarning, stacklevel=2)
        return out

    # -- reporting -----------------------------------------------------------

    def summary(self):
        """Human-readable summary: k, T, ν, and the CV table (if any)."""
        lines = [f'LowRankNoise: {self.n_voxels} voxels, k = {self.k}, '
                 f'temperature T = {self.temperature:g}, '
                 f'dof = {"Gaussian" if self.dof is None else f"{self.dof:.3g}"}']
        common = (self.factors ** 2).sum(1)
        lines.append(f'  mean shared (factor) variance fraction = '
                     f'{np.mean(common / (common + self.psi)):.3f}; '
                     f'psi at floor: {np.mean(self.psi <= getattr(self, "psi_floor", 0) + 1e-12):.1%}')
        if self.cv_scores_ is not None:
            cv = self.cv_scores_
            info = self.cv_info_
            lines.append(f'  chosen by {info["n_groups"]}-group nested CV on '
                         f'{info["n_trials"]} trials, {info["n_stimuli"]} grid points; '
                         f'inner parameters: {info["inner_parameters"]}')
            lines.append('  CV score = mean held-out log p(true stimulus) − log(1/n_stimuli), nats/trial')
            best_T = cv.idxmax(axis=1)
            tab = pd.DataFrame({'best_T': best_T, 'score_at_best_T': cv.max(axis=1)})
            if 1.0 in cv.columns:
                tab['score_at_T=1'] = cv[1.0]
            lines.append(tab.to_string(float_format=lambda x: f'{x:.4g}'))
        return '\n'.join(lines)

    def __repr__(self):
        return (f'LowRankNoise(n_voxels={self.n_voxels}, k={self.k}, '
                f'temperature={self.temperature:g}, dof={self.dof})')


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------

def _subset(x, idx):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.iloc[idx]
    return np.asarray(x)[idx]


def _grid_predictions(model, grid, parameters, weights=None):
    """Model prediction for every grid stimulus, (n_stimuli, V) float64.

    Same code path as ``EncodingModel.get_stimulus_pdf`` (each grid point is
    a separate batch, so HRF models are not convolved across the grid).
    """
    return model.predict_stimulus_grid(grid, parameters=parameters, weights=weights)


def fit_lowrank_omega(residuals, k=40, dof='fit', psi_floor=0.02):
    """Functional / legacy path: dense ``(omega, dof)`` of a low-rank fit.

    Drop-in replacement for the ``(omega, dof)`` returned by
    ``ResidualFitter.fit`` (pass them as ``omega=`` / ``dof=``). Prefer
    :class:`LowRankNoise` for large V (no dense matrix) and for choosing
    ``k`` and the temperature by cross-validation.
    """
    return LowRankNoise.from_residuals(residuals, k=k, dof=dof,
                                       psi_floor=psi_floor).to_dense()
