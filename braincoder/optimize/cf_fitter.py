"""Grid search (and optional gradient refinement) for connective-field models.

The grid follows Haak et al. (2013): every source vertex is a candidate CF
centre, crossed with a small set of sizes sigma. For each candidate the
pooled source signal ``p_c = X w_c`` is computed once; a target's best
candidate is the one whose pooled signal correlates most with it. Because
``baseline`` and ``amplitude`` enter linearly, least squares with a free
baseline and a positive amplitude selects exactly the same candidate
(R^2 = r^2 for r > 0), so the correlation search *is* the least-squares grid
fit; amplitude and baseline then follow in closed form.

Cost: one ``(n_samples x n_sources) @ (n_sources x n_candidates)`` product
for the candidate signals, then ``(n_targets x n_samples) @ (n_samples x
n_candidates)`` in target batches, all with ``keras.ops`` (TF / JAX / torch,
CPU or GPU). Only the per-target winner is kept, so memory is bounded by the
batch size.
"""
import logging

import numpy as np
import pandas as pd
from keras import ops

from ..utils import format_data, get_rsq

logger = logging.getLogger(__name__)


def exclusion_from_distances(distances, radius):
    """Leave-neighbourhood-out mask from target-to-source distances.

    Parameters
    ----------
    distances : array (n_targets, n_sources)
        Geodesic distance from each target vertex to each source vertex
        (``inf`` where none). Targets that are not near any source get an
        empty row.
    radius : float
        Sources closer than ``radius`` to a target are excluded from all of
        that target's candidate fields, and cannot be its centre.

    Returns
    -------
    scipy.sparse.csr_matrix (n_targets, n_sources), boolean
    """
    from scipy import sparse
    distances = np.asarray(distances)
    return sparse.csr_matrix(np.isfinite(distances) & (distances < radius))


class ConnectiveFieldFitter:
    """Fit :class:`braincoder.models.GaussianConnectiveField` models.

    Parameters
    ----------
    model : GaussianConnectiveField
        Holds the source distance matrix, truncation and normalisation.
    data : DataFrame or array (n_samples, n_targets)
        Target signals.
    paradigm : DataFrame or array (n_samples, n_sources), optional
        Source signals; defaults to ``model.paradigm``. Pre-process sources
        as you see fit before passing them (e.g. z-score, remove the
        source-region mean); the fitter does not.
    """

    def __init__(self, model, data, paradigm=None):
        self.model = model
        self.data = format_data(data).astype(np.float32)
        self.paradigm = model.get_paradigm(paradigm)
        if self.paradigm.shape[0] != self.data.shape[0]:
            raise ValueError(f'paradigm has {self.paradigm.shape[0]} samples, '
                             f'data has {self.data.shape[0]}')
        if self.paradigm.shape[1] != model.n_sources:
            raise ValueError(f'paradigm has {self.paradigm.shape[1]} sources, '
                             f'model has {model.n_sources}')

    # ---------------------------------------------------------- candidates
    def _candidates(self, sigmas, centers):
        n_src = self.model.n_sources
        centers = np.arange(n_src) if centers is None else np.asarray(centers, dtype=int)
        sigmas = np.atleast_1d(np.asarray(sigmas, dtype=np.float32))
        cand_centre = np.tile(centers, len(sigmas))
        cand_sigma = np.repeat(sigmas, len(centers))
        return cand_centre, cand_sigma

    def _candidate_signals(self, X, cand_centre, cand_sigma, batch_size):
        """Raw (unnormalised) pooled signals ``X @ w_c`` for every candidate,
        plus each candidate's normalisation factor ``k_c``."""
        P, K = [], []
        D = self.model._D
        for i0 in range(0, len(cand_centre), batch_size):
            c = ops.convert_to_tensor(cand_centre[i0:i0 + batch_size].astype(np.int32))
            s = ops.convert_to_tensor(cand_sigma[i0:i0 + batch_size])[:, None]
            w = self.model._gaussian_weights(ops.take(D, c, axis=0), s, normalize=None)
            P.append(ops.matmul(X, ops.transpose(w)))
            if self.model.normalize == 'sum':
                K.append(1.0 / ops.maximum(ops.sum(w, axis=1), 1e-12))
            elif self.model.normalize == 'density':
                K.append(1.0 / (2.0 * np.pi * ops.square(s[:, 0])))
            else:
                K.append(ops.ones_like(s[:, 0]))
        return ops.concatenate(P, axis=1), ops.convert_to_numpy(ops.concatenate(K, axis=0))

    @staticmethod
    def _standardise(A):
        mean = ops.mean(A, axis=0)
        sd = ops.std(A, axis=0)
        valid = sd > 1e-8 * (ops.max(sd) + 1e-30)
        Z = (A - mean[None, :]) / ops.where(valid, sd, ops.ones_like(sd))[None, :]
        return Z, mean, sd, valid

    # ---------------------------------------------------------- grid fit
    def fit_grid(self, sigmas, centers=None, exclude=None, positive=True,
                 candidate_batch_size=2048, target_batch_size=1024,
                 progressbar=True):
        """Grid search over CF centre x size.

        Parameters
        ----------
        sigmas : sequence of float
            Candidate CF sizes (distance-matrix units, usually mm).
        centers : array of int, optional
            Candidate centres (source indices). Default: every source.
        exclude : sparse or dense bool (n_targets, n_sources), or dict, optional
            Leave-neighbourhood-out: for each target, the sources to drop
            from every candidate pool (they also cannot be its centre).
            Needed when a target is itself in, or next to, the source region
            (e.g. V1 targets with V1 sources), where a target's own signal
            or its smoothed neighbours would otherwise win trivially. A dict
            maps a target's column position to an array of source indices.
            See :func:`exclusion_from_distances`.
        positive : bool
            Maximise r (positive coupling, as Haak et al.). ``False``
            maximises |r| (allows negative amplitudes).
        candidate_batch_size, target_batch_size : int
            Memory bounds.

        Returns
        -------
        DataFrame (n_targets, 4)
            ``center_index, sigma, baseline, amplitude``. The fit quality
            of the search (with exclusions applied) is stored in
            ``self.grid_r_`` (correlation) and ``self.grid_rsq_`` (r^2).
        """
        X = ops.convert_to_tensor(self.paradigm.values.astype(np.float32))
        Y = ops.convert_to_tensor(self.data.values.astype(np.float32))
        n_t, n_v = self.data.shape
        cand_centre, cand_sigma = self._candidates(sigmas, centers)
        n_c = len(cand_centre)

        P, K = self._candidate_signals(X, cand_centre, cand_sigma, candidate_batch_size)
        Zp, p_mean, p_sd, p_valid = self._standardise(P)
        Zy, y_mean, y_sd, y_valid = self._standardise(Y)
        neg_inf = ops.full((1, n_c), -np.inf)

        excl_rows = self._exclusion_rows(exclude, n_v)

        best = np.zeros(n_v, dtype=np.int64)
        r_search = np.full(n_v, np.nan, dtype=np.float32)
        r_full = np.full(n_v, np.nan, dtype=np.float32)

        batches = range(0, n_v, target_batch_size)
        if progressbar:
            from tqdm.auto import tqdm
            batches = tqdm(batches, desc='CF grid')

        for i0 in batches:
            i1 = min(i0 + target_batch_size, n_v)
            r = ops.matmul(ops.transpose(Zy[:, i0:i1]), Zp) / n_t        # (v, c)
            r = ops.where(p_valid[None, :], r, neg_inf)
            score = r if positive else ops.where(p_valid[None, :], ops.abs(r), neg_inf)
            b = ops.convert_to_numpy(ops.argmax(score, axis=1))
            rb = ops.convert_to_numpy(ops.take_along_axis(r, ops.convert_to_tensor(b[:, None]), axis=1))[:, 0]
            best[i0:i1], r_search[i0:i1], r_full[i0:i1] = b, rb, rb

            todo = [j for j in range(i0, i1) if j in excl_rows]
            if todo:
                r_np = ops.convert_to_numpy(r)
                for j in todo:
                    b_j, r_j = self._fit_excluded(
                        j, r_np[j - i0], excl_rows[j], X, Y, P, cand_centre, cand_sigma,
                        positive)
                    best[j], r_search[j], r_full[j] = b_j, r_j, r_np[j - i0, b_j]

        # closed-form amplitude / baseline on the model's (normalised, full)
        # pooled signal of the winning candidate
        p_sd_np, p_mean_np = ops.convert_to_numpy(p_sd), ops.convert_to_numpy(p_mean)
        y_sd_np, y_mean_np = ops.convert_to_numpy(y_sd), ops.convert_to_numpy(y_mean)
        k = K[best]
        with np.errstate(invalid='ignore', divide='ignore'):
            amplitude = r_full * y_sd_np / (k * p_sd_np[best])
        baseline = y_mean_np - amplitude * k * p_mean_np[best]

        y_valid = ops.convert_to_numpy(y_valid)
        r_search[~y_valid] = np.nan
        pars = pd.DataFrame({'center_index': cand_centre[best].astype(np.float32),
                             'sigma': cand_sigma[best],
                             'baseline': np.where(y_valid, baseline, np.nan),
                             'amplitude': np.where(y_valid, amplitude, np.nan)},
                            index=self.data.columns).astype(np.float32)
        pars.index.name = self.data.columns.name
        self.grid_parameters = pars
        self.grid_r_ = pd.Series(r_search, index=self.data.columns, name='r')
        self.grid_rsq_ = (self.grid_r_ ** 2).rename('r2')
        self._grid = dict(cand_centre=cand_centre, cand_sigma=cand_sigma, best=best, K=K,
                          sigmas=np.atleast_1d(sigmas), centers=centers)
        return pars

    @staticmethod
    def _exclusion_rows(exclude, n_v):
        if exclude is None:
            return {}
        if isinstance(exclude, dict):
            return {int(j): np.asarray(e, dtype=int) for j, e in exclude.items() if len(e)}
        from scipy import sparse
        E = sparse.csr_matrix(exclude, dtype=bool)
        if E.shape[0] != n_v:
            raise ValueError(f'exclude has {E.shape[0]} rows, data has {n_v} targets')
        return {j: E.indices[E.indptr[j]:E.indptr[j + 1]]
                for j in range(n_v) if E.indptr[j + 1] > E.indptr[j]}

    def _fit_excluded(self, j, r_row, E, X, Y, P, cand_centre, cand_sigma, positive):
        """Re-score target j with sources E removed from every pool."""
        D = self.model.distance_matrix
        dE = D[np.ix_(E, cand_centre)]                                    # (|E|, c)
        reach = np.isfinite(dE)
        if self.model.truncate is not None:
            reach &= dE <= self.model.truncate * cand_sigma[None, :]
        rel = np.flatnonzero(reach.any(0))
        score = np.array(r_row, dtype=np.float64)
        if len(rel):
            d = ops.convert_to_tensor(np.where(reach[:, rel], dE[:, rel], np.inf).astype(np.float32))
            wE = self.model._gaussian_weights(
                ops.transpose(d), ops.convert_to_tensor(cand_sigma[rel])[:, None],
                normalize=None)                                            # (rel, |E|)
            xE = ops.take(X, ops.convert_to_tensor(E.astype(np.int32)), axis=1)  # (t, |E|)
            Pr = ops.take(P, ops.convert_to_tensor(rel.astype(np.int32)), axis=1) - \
                ops.matmul(xE, ops.transpose(wE))
            Pr = ops.convert_to_numpy(Pr).astype(np.float64)
            y = ops.convert_to_numpy(Y[:, j]).astype(np.float64)
            Pr -= Pr.mean(0)
            y = y - y.mean()
            with np.errstate(invalid='ignore', divide='ignore'):
                rr = (y @ Pr) / np.sqrt((Pr ** 2).sum(0) * (y ** 2).sum())
            score[rel] = np.where(np.isfinite(rr), rr, -np.inf)
        score[np.isin(cand_centre, E)] = -np.inf
        b = int(np.argmax(score if positive else np.where(np.isfinite(score), np.abs(score), -np.inf)))
        return b, float(score[b])

    # ---------------------------------------------------------- refinement
    def refine(self, parameters=None, fixed_pars=('center_index',), **fit_kwargs):
        """Gradient refinement of the continuous parameters.

        Runs :class:`braincoder.optimize.ParameterFitter` from the grid
        solution with ``center_index`` held fixed (it is discrete). By
        default ``sigma``, ``baseline`` and ``amplitude`` are optimised by
        least squares; pass e.g. ``fixed_pars=('center_index', 'sigma')`` to
        keep the grid sigma. ``fit_kwargs`` go to ``ParameterFitter.fit``.
        """
        from .parameter_fitter import ParameterFitter
        if parameters is None:
            parameters = self.grid_parameters
        ok = parameters.notna().all(axis=1)
        fitter = ParameterFitter(self.model, self.data.loc[:, ok], self.paradigm)
        refined = fitter.fit(init_pars=parameters.loc[ok], fixed_pars=list(fixed_pars),
                             **fit_kwargs)
        out = parameters.copy()
        out.loc[ok] = refined.values
        self.estimated_parameters = out
        return out

    # ---------------------------------------------------------- evaluation
    def get_predictions(self, parameters=None, batch_size=1024):
        if parameters is None:
            parameters = getattr(self, 'estimated_parameters', self.grid_parameters)
        return self.model.predict(self.paradigm, parameters.fillna(0), batch_size=batch_size)

    def get_rsq(self, parameters=None, batch_size=1024):
        return get_rsq(self.data, self.get_predictions(parameters, batch_size))

    def fit_cv(self, groups, sigmas, centers=None, exclude=None, positive=True,
               progressbar=False, **grid_kwargs):
        """Cross-validated connective fields (leave-one-group-out).

        For each fold the grid fit (centre, sigma, baseline, amplitude) is
        estimated on the training samples and used to predict the held-out
        samples. Group by run, so that temporal autocorrelation within a run
        does not leak between train and test.

        Returns
        -------
        DataFrame (n_targets, 2)
            ``cv_r2``: 1 - SS_res / SS_tot over the concatenated out-of-fold
            predictions; ``cv_r``: their correlation with the data.
            Per-fold parameters are kept in ``self.cv_parameters`` (dict).
        """
        groups = np.asarray(groups)
        if len(groups) != len(self.data):
            raise ValueError('groups must have one entry per sample')
        pred = np.zeros(self.data.shape, dtype=np.float64)
        self.cv_parameters = {}
        X_all = self.paradigm.values.astype(np.float32)
        for g in np.unique(groups):
            train, test = groups != g, groups == g
            sub = ConnectiveFieldFitter(self.model, self.data.loc[train], self.paradigm.loc[train])
            pars = sub.fit_grid(sigmas, centers=centers, exclude=exclude, positive=positive,
                                progressbar=progressbar, **grid_kwargs)
            self.cv_parameters[g] = pars
            pred[test] = self._predict_candidates(X_all[test], pars, sub._grid)
        y = self.data.values.astype(np.float64)
        ss_tot = ((y - y.mean(0)) ** 2).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            cv_r2 = 1 - ((y - pred) ** 2).sum(0) / ss_tot
            yc, pc = y - y.mean(0), pred - pred.mean(0)
            cv_r = (yc * pc).sum(0) / np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(0))
        return pd.DataFrame({'cv_r2': cv_r2, 'cv_r': cv_r}, index=self.data.columns)

    def _predict_candidates(self, X, pars, grid):
        """Predict via the winning candidates' pooled signals (no per-target
        weight matrices: cheap for many targets)."""
        cc, cs, best, K = grid['cand_centre'], grid['cand_sigma'], grid['best'], grid['K']
        ub, inv = np.unique(best, return_inverse=True)
        Xt = ops.convert_to_tensor(X)
        P, _ = self._candidate_signals(Xt, cc[ub], cs[ub], 4096)
        P = ops.convert_to_numpy(P)[:, inv] * K[best][None, :]
        return P * pars['amplitude'].values[None, :] + pars['baseline'].values[None, :]
