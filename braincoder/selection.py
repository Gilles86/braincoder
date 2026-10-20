"""Label-permutation voxel selection.

Which voxels carry stimulus information? Fit the encoding model to the real
data, then refit it ``K`` times after shuffling the stimulus labels *within*
exchangeability groups (typically runs), with the identical fitting
procedure. The shuffled fits give a null distribution of the fit statistic
(R²) that includes everything the fitting procedure does to noise: grid
search, gradient descent, overfitting by flexible models. Voxels whose real
R² exceeds that null are selected, with Benjamini-Hochberg FDR control within
a user-given family (e.g. one ROI).

Two ways to turn the null into p-values:

``'pooled'`` (default)
    One null pool over all voxels and permutations::

        p(v) = (1 + #{null >= R²(v)}) / (1 + K·V)

    Valid when the null is exchangeable across voxels (the same distribution
    in every voxel). Then ``K`` can be small (5-20): the pool has ``K·V``
    values. :meth:`PermutationSelectionResult.exchangeability_report` checks
    the assumption.
    Granularity: the smallest pooled p is ``1 / (1 + K·V)``, so BH at level
    ``q`` can only reject once roughly ``1 / (q·K)`` voxels exceed *every*
    null value (4 voxels for K = 5, q = .05). With small K the rule is
    therefore conservative for very sparse signals: it cannot select one or
    two isolated voxels. Raise K if that matters.
``'voxelwise'``
    Each voxel against its own ``K`` null values,
    ``p(v) = (1 + #{null(v) >= R²(v)}) / (1 + K)``. Needs no exchangeability
    but needs a large ``K`` (the smallest possible p is ``1 / (1 + K)``).

Example
-------
>>> from braincoder.selection import PermutationSelector
>>> sel = PermutationSelector(model, n_permutations=5, groups=runs,
...                           q=0.05, seed=(subject, fold))
>>> res = sel.fit(data, paradigm, fit_kwargs=dict(grid=(mus, sds, [1.], [0.])))
>>> res.mask.sum(), res.exchangeability_report()

``model`` can be a braincoder :class:`~braincoder.models.EncodingModel`
(fitted with :func:`make_r2_fitter`'s default procedure) or any callable
``fit_fn(data, paradigm, **fit_kwargs) -> per-voxel R²``, so the selection
uses exactly the procedure you use for the real analysis.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['PermutationSelector', 'PermutationSelectionResult',
           'make_r2_fitter', 'permutation_indices', 'pooled_pvalues',
           'voxelwise_pvalues', 'bh_qvalues']


# --------------------------------------------------------------------------
# p-values and FDR

def bh_qvalues(p):
    """Benjamini-Hochberg q-values (monotone). NaN in, NaN out (not tested)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def pooled_pvalues(r2, null_values):
    """``p = (1 + #{null >= r2}) / (1 + n_null)`` against one pooled null.

    ``null_values`` is any array of null statistics (flattened; non-finite
    values are dropped). Vectorised with ``searchsorted``.
    """
    pool = np.asarray(null_values, dtype=float).ravel()
    pool = np.sort(pool[np.isfinite(pool)])
    n = len(pool)
    r2 = np.asarray(r2, dtype=float)
    n_ge = n - np.searchsorted(pool, r2, side='left')
    p = (1.0 + n_ge) / (1.0 + n)
    return np.where(np.isfinite(r2), p, np.nan)


def voxelwise_pvalues(r2, null_r2):
    """Per-voxel p against its own null: ``(1 + #{null[:, v] >= r2[v]}) / (1 + K)``."""
    r2 = np.asarray(r2, dtype=float)
    null_r2 = np.asarray(null_r2, dtype=float)
    K = np.isfinite(null_r2).sum(0)
    n_ge = (null_r2 >= r2[None, :]).sum(0)
    p = (1.0 + n_ge) / (1.0 + K)
    return np.where(np.isfinite(r2), p, np.nan)


# --------------------------------------------------------------------------
# shuffling

def _seed_sequence(seed, k):
    if seed is None:
        raise ValueError('seed must be set (int or tuple of ints) for reproducible shuffles')
    base = list(seed) if isinstance(seed, (tuple, list, np.ndarray)) else [seed]
    return np.random.SeedSequence([int(s) for s in base] + [int(k)])


def permutation_indices(n_rows, groups=None, n_permutations=5, seed=0, start=0):
    """Row permutations that only move rows within their group.

    Permutation ``k`` is drawn from ``SeedSequence([*seed, k])``, so it does
    not depend on ``n_permutations``: asking for more permutations later
    extends the set without changing the first ones.

    Returns
    -------
    np.ndarray, shape ``(n_permutations, n_rows)``
        ``perms[k][i]`` is the row whose stimulus row ``i`` receives in
        permutation ``k`` (i.e. the shuffled paradigm is ``paradigm[perms[k]]``).
    """
    if groups is None:
        groups = np.zeros(n_rows, dtype=int)
    groups = np.asarray(groups)
    if len(groups) != n_rows:
        raise ValueError(f'groups has length {len(groups)}, data has {n_rows} rows')
    _, codes = np.unique(groups, return_inverse=True)
    members = [np.where(codes == g)[0] for g in range(codes.max() + 1)]
    perms = np.empty((n_permutations, n_rows), dtype=np.int64)
    for i, k in enumerate(range(start, start + n_permutations)):
        rng = np.random.default_rng(_seed_sequence(seed, k))
        perm = np.arange(n_rows)
        for m in members:
            perm[m] = m[rng.permutation(len(m))]
        perms[i] = perm
    return perms


def _resolve_groups(groups, data):
    if groups is None:
        return None
    names = [groups] if isinstance(groups, str) else (
        list(groups) if isinstance(groups, (list, tuple)) else None)
    idx = data.index
    if names is not None and len(names) > 0 and all(
            isinstance(n, str) and n in (idx.names or []) for n in names):
        # index level name(s)
        vals = [np.asarray(idx.get_level_values(n)).astype(str) for n in names]
        return np.array(['|'.join(v) for v in zip(*vals)])
    return np.asarray(groups)


def _take_rows(paradigm, perm):
    """Paradigm with its rows reordered by ``perm`` but its index kept."""
    if isinstance(paradigm, pd.DataFrame):
        return pd.DataFrame(paradigm.values[perm], index=paradigm.index,
                            columns=paradigm.columns)
    if isinstance(paradigm, pd.Series):
        return pd.Series(paradigm.values[perm], index=paradigm.index,
                         name=paradigm.name)
    return np.asarray(paradigm)[perm]


# --------------------------------------------------------------------------
# default fitter

def make_r2_fitter(model, **default_fit_kwargs):
    """A ``fit_fn(data, paradigm, **fit_kwargs) -> R²`` for a braincoder model.

    Procedure (all optional via ``fit_kwargs``):

    * ``grid`` -- tuple of per-parameter grids for
      :meth:`ParameterFitter.fit_grid` (with ``use_correlation_cost=True``),
      followed by ``refine_baseline_and_amplitude(n_iterations=refine)``
      (``refine``, default 2). Without ``grid`` the model's
      ``get_init_pars`` is used.
    * ``fixed_pars_first`` -- if given, a first gradient-descent pass with
      these parameters fixed (learning rate ``learning_rate_first``).
    * remaining kwargs go to :meth:`ParameterFitter.fit` (e.g.
      ``learning_rate``, ``max_n_iterations``, ``r2_atol``).

    For anything more specific, pass your own callable to
    :class:`PermutationSelector` instead.
    """
    from .optimize import ParameterFitter
    from .utils import get_rsq

    def fit_fn(data, paradigm, **fit_kwargs):
        kw = dict(default_fit_kwargs)
        kw.update(fit_kwargs)
        grid = kw.pop('grid', None)
        refine = kw.pop('refine', 2)
        fixed_first = kw.pop('fixed_pars_first', None)
        lr_first = kw.pop('learning_rate_first', kw.get('learning_rate', 0.01))
        kw.setdefault('progressbar', False)
        fitter = ParameterFitter(model, data, paradigm)
        init = None
        if grid is not None:
            init = fitter.fit_grid(*grid, use_correlation_cost=True)
            if refine:
                init = fitter.refine_baseline_and_amplitude(init, n_iterations=refine)
        if fixed_first:
            kw1 = dict(kw, learning_rate=lr_first)
            init = fitter.fit(init_pars=init, fixed_pars=list(fixed_first), **kw1)
        pars = fitter.fit(init_pars=init, **kw)
        pred = model.predict(parameters=pars, paradigm=paradigm)
        return get_rsq(data, pred)

    return fit_fn


# --------------------------------------------------------------------------
# result

@dataclass
class PermutationSelectionResult:
    """Outcome of :meth:`PermutationSelector.fit`.

    Attributes
    ----------
    r2 : pd.Series (V,)
        Real-data statistic per voxel.
    null_r2 : pd.DataFrame (K, V)
        Statistic of each within-group label shuffle (rows = permutations).
    p, q : pd.Series (V,)
        p-values (pooled or voxelwise) and BH q-values within each
        ``bh_groups`` family. NaN = not tested.
    mask : pd.Series of bool (V,)
        ``q <= q_thresh``.
    permutations : np.ndarray (K, n_rows)
        The row permutations used (``paradigm[perm]``).
    """
    r2: pd.Series
    null_r2: pd.DataFrame
    p: pd.Series
    q: pd.Series
    mask: pd.Series
    q_thresh: float
    method: str
    permutations: np.ndarray
    bh_groups: pd.Series | None = None
    pool_mask: np.ndarray | None = None
    pool_within_family: bool = False
    info: dict = field(default_factory=dict)

    @property
    def n_selected(self):
        return int(self.mask.sum())

    @property
    def p_voxelwise(self):
        """Unpooled p: each voxel against its own K null values,
        ``(1 + #{null(v) >= R²(v)}) / (1 + K)``. Needs no exchangeability,
        but is bounded below by ``1 / (1 + K)``."""
        p = voxelwise_pvalues(self.r2.values, self.null_r2.values)
        if self.bh_groups is not None:
            p = np.where(pd.notnull(self.bh_groups.values), p, np.nan)
        return pd.Series(p, index=self.r2.index)

    def selected(self):
        """Index of the selected voxels."""
        return self.r2.index[self.mask.values]

    def summary(self):
        """Per BH family: number tested, selected, min selected R²."""
        fam = (self.bh_groups if self.bh_groups is not None
               else pd.Series('all', index=self.r2.index))
        rows = []
        for g in pd.unique(fam.dropna()):
            s = (fam == g).values & np.isfinite(self.p.values)
            sel = s & self.mask.values
            rows.append(dict(family=g, n_tested=int(s.sum()), n_selected=int(sel.sum()),
                             min_selected_r2=float(self.r2.values[sel].min()) if sel.any() else np.nan))
        return pd.DataFrame(rows)

    def exchangeability_report(self, warn=True, r_tol=0.1, var_ratio_tol=1.5,
                               by_family=False):
        """Is the null exchangeable across voxels (the pooled-p assumption)?

        * ``split_half_r``: Pearson r across voxels between each voxel's mean
          null R² in the even and in the odd permutations. If the null were
          voxel-specific, voxels with a high null in one half would have a
          high null in the other (r > 0). Under exchangeability r ≈ 0.
        * ``var_ratio``: variance across voxels of the per-voxel null mean,
          divided by its expectation if every voxel drew its K values from
          the pooled null (``var(pool) / K``). ≈ 1 under exchangeability,
          > 1 if voxels differ.
        * ``excess_exceed_frac``: fraction of voxels whose count of null
          values above the pooled 95th percentile exceeds the Binomial(K, .05)
          99th percentile (≈ ≤ 0.01 under exchangeability; coarse for small K).

        Spatially smooth data reduce the effective number of independent
        voxels, so ``split_half_p`` (from the Fisher z with n = V) is
        optimistic; the warning therefore also requires ``r > r_tol``.
        Needs K >= 2 (K >= 4 for a meaningful split half).

        With ``by_family=True`` the report is computed separately within each
        BH family and returned as a DataFrame (one row per family).
        """
        if by_family:
            fam = (self.bh_groups if self.bh_groups is not None
                   else pd.Series('all', index=self.r2.index))
            rows = []
            for g in pd.unique(fam.dropna()):
                m = (fam == g).values
                rows.append(dict(family=g, **self._exch(m, warn, r_tol, var_ratio_tol)))
            return pd.DataFrame(rows)
        m = np.ones(len(self.r2), bool)
        return self._exch(m, warn, r_tol, var_ratio_tol)

    def _exch(self, voxel_mask, warn, r_tol, var_ratio_tol):
        from scipy import stats
        null = self.null_r2.values
        m = voxel_mask.copy()
        if self.pool_mask is not None:
            m &= self.pool_mask
        null = null[:, m]
        null = null[:, np.all(np.isfinite(null), 0)]
        K, V = null.shape
        rep = dict(K=int(K), n_voxels=int(V))
        if K < 2 or V < 3:
            rep.update(split_half_r=np.nan, split_half_p=np.nan, var_ratio=np.nan,
                       excess_exceed_frac=np.nan, exchangeable=None)
            return rep
        a, b = null[0::2].mean(0), null[1::2].mean(0)
        r = float(np.corrcoef(a, b)[0, 1])
        z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(max(V - 3, 1))
        p_r = float(stats.norm.sf(z))
        var_ratio = float(null.mean(0).var() / (null.var() / K))
        thr = np.quantile(null, 0.95)
        n_exc = (null > thr).sum(0)
        crit = stats.binom.ppf(0.99, K, 0.05)
        excess = float((n_exc > crit).mean())
        ok = not ((r > r_tol and p_r < 0.01) or var_ratio > var_ratio_tol)
        rep.update(split_half_r=r, split_half_p=p_r, var_ratio=var_ratio,
                   excess_exceed_frac=excess, binom_crit=float(crit),
                   null_q95=float(thr), exchangeable=bool(ok))
        if warn and not ok:
            warnings.warn(
                f'Permutation null does not look exchangeable across voxels '
                f'(split-half r = {r:.3f}, var ratio = {var_ratio:.2f}); pooled p-values '
                f'may be miscalibrated -- consider method="voxelwise" with a larger K.',
                RuntimeWarning, stacklevel=2)
        return rep

    def calibration(self, q=None):
        """Leave-one-permutation-out false-positive check.

        Treat each permutation as if it were the real data, test it against
        the other K - 1 (same method, same BH families) and count discoveries.
        Under the complete null BH controls the family-wise rate, so
        ``any_discovery_rate`` should be ≲ q.
        """
        q = self.q_thresh if q is None else q
        null = self.null_r2.values
        K = null.shape[0]
        n_disc = []
        for k in range(K):
            others = np.delete(null, k, axis=0)
            _, qk = _pq(null[k], others, self.method,
                        None if self.bh_groups is None else self.bh_groups.values,
                        self.pool_mask, self.pool_within_family)
            n_disc.append(int(np.nansum(qk <= q)))
        n_disc = np.asarray(n_disc)
        return dict(any_discovery_rate=float((n_disc > 0).mean()),
                    mean_discovery_frac=float(n_disc.mean() / max(np.isfinite(self.p).sum(), 1)),
                    n_discoveries=n_disc.tolist())


def _pq(r2, null, method, bh_groups, pool_mask, pool_within_family=False):
    if method == 'pooled':
        pm = np.ones(null.shape[1], bool) if pool_mask is None else pool_mask
        if pool_within_family and bh_groups is not None:
            fam = np.asarray(bh_groups, dtype=object)
            p = np.full(np.shape(r2), np.nan)
            for g in pd.unique(fam[pd.notnull(fam)]):
                s = fam == g
                p[s] = pooled_pvalues(r2[s], null[:, s & pm])
        else:
            p = pooled_pvalues(r2, null[:, pm])
    elif method == 'voxelwise':
        p = voxelwise_pvalues(r2, null)
    else:
        raise ValueError(f"method must be 'pooled' or 'voxelwise', got {method!r}")
    if bh_groups is None:
        q = bh_qvalues(p)
    else:
        fam = np.asarray(bh_groups, dtype=object)
        tested = pd.notnull(fam)
        p = np.where(tested, p, np.nan)
        q = np.full(p.shape, np.nan)
        for g in pd.unique(fam[tested]):
            s = fam == g
            q[s] = bh_qvalues(p[s])
    return p, q


# --------------------------------------------------------------------------
# selector

class PermutationSelector:
    """Select voxels whose fit beats a within-group label-permutation null.

    Parameters
    ----------
    fitter : EncodingModel or callable
        Either a braincoder model (fitted via :func:`make_r2_fitter`), or
        ``fit_fn(data, paradigm, **fit_kwargs)`` returning one statistic per
        voxel (array-like of length V, higher = better fit; normally R²).
        The *same* callable fits the real and every shuffled paradigm.
    n_permutations : int
        K, number of label shuffles.
    groups : array-like, str or list of str, optional
        Exchangeability blocks (e.g. run labels, one per data row), or the
        name(s) of index level(s) of ``data`` (e.g. ``['session', 'run']``).
        Stimulus rows are shuffled only within a group. ``None`` = one group.
    q : float
        BH threshold for ``mask``.
    seed : int or tuple of ints
        Permutation k uses ``SeedSequence([*seed, k])``.
    method : {'pooled', 'voxelwise'}
    stack : bool
        If True, fit the real data and all K shuffles in ONE call: shuffling
        the stimulus rows is equivalent to applying the inverse permutation to
        the data rows, so the K + 1 versions are concatenated as extra voxels
        against the unshuffled paradigm. Much faster on a GPU for small voxel
        sets, and real and null share the exact same optimiser run (including
        early stopping). Only valid for models whose prediction for a row
        depends on that row's stimulus only (trial-wise / single-trial-beta
        models), not for models with temporal structure (HRF convolution).
    """

    def __init__(self, fitter, n_permutations=5, groups=None, q=0.05, seed=0,
                 method='pooled', stack=False):
        if callable(fitter) and not hasattr(fitter, 'predict'):
            self.fit_fn = fitter
        else:
            self.fit_fn = make_r2_fitter(fitter)
        if method not in ('pooled', 'voxelwise'):
            raise ValueError(method)
        self.n_permutations = int(n_permutations)
        self.groups = groups
        self.q = q
        self.seed = seed
        self.method = method
        self.stack = stack

    def _as_series(self, r2, columns):
        r2 = np.asarray(r2, dtype=float).ravel()
        if len(r2) != len(columns):
            raise ValueError(f'fit function returned {len(r2)} values for {len(columns)} voxels')
        return pd.Series(r2, index=columns)

    def fit(self, data, paradigm, fit_kwargs=None, bh_groups=None, pool_mask=None,
            pool_within_family=False):
        """Real fit + K shuffled fits; p, q and mask.

        Parameters
        ----------
        data : pd.DataFrame (n_rows, V)
        paradigm : DataFrame / Series / array with n_rows rows
        fit_kwargs : dict, passed to the fit function every time
        bh_groups : array-like (V,), optional
            BH family per voxel (e.g. ROI labels); BH is applied within each
            family; voxels labelled None/NaN are not tested (q = NaN). Default:
            all voxels form one family.
        pool_mask : bool array (V,), optional
            Voxels whose null values enter the pooled null (default: all).
        pool_within_family : bool
            ``method='pooled'`` only: build a separate pooled null for each
            ``bh_groups`` family from that family's own voxels (e.g. one null
            per ROI) instead of one pool over all voxels.
        """
        fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
        data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        n_rows, V = data.shape
        if len(paradigm) != n_rows:
            raise ValueError('paradigm and data have different numbers of rows')
        groups = _resolve_groups(self.groups, data)
        perms = permutation_indices(n_rows, groups, self.n_permutations, self.seed)
        K = len(perms)

        if self.stack:
            # Fitting Y against X[pi] == fitting Y[pi^-1] against X (row-wise models).
            blocks = [data.values] + [data.values[np.argsort(pm)] for pm in perms]
            stacked = pd.DataFrame(np.concatenate(blocks, axis=1), index=data.index,
                                   columns=np.arange((K + 1) * V))
            allr2 = np.asarray(self.fit_fn(stacked, paradigm, **fit_kwargs),
                               dtype=float).ravel().reshape(K + 1, V)
            r2 = pd.Series(allr2[0], index=data.columns)
            null = allr2[1:]
        else:
            r2 = self._as_series(self.fit_fn(data, paradigm, **fit_kwargs), data.columns)
            null = np.empty((K, V))
            for k, pm in enumerate(perms):
                null[k] = self._as_series(
                    self.fit_fn(data, _take_rows(paradigm, pm), **fit_kwargs),
                    data.columns).values
                logger.info('permutation %d/%d done', k + 1, K)

        bh = None if bh_groups is None else pd.Series(np.asarray(bh_groups, dtype=object),
                                                      index=data.columns)
        pm_ = None if pool_mask is None else np.asarray(pool_mask, dtype=bool)
        p, q = _pq(r2.values, null, self.method, None if bh is None else bh.values, pm_,
                   pool_within_family)
        null_df = pd.DataFrame(null, columns=data.columns,
                               index=pd.Index(np.arange(K), name='permutation'))
        q_s = pd.Series(q, index=data.columns)
        return PermutationSelectionResult(
            r2=r2, null_r2=null_df, p=pd.Series(p, index=data.columns), q=q_s,
            mask=(q_s <= self.q).fillna(False).astype(bool), q_thresh=self.q,
            method=self.method, permutations=perms, bh_groups=bh, pool_mask=pm_,
            pool_within_family=bool(pool_within_family),
            info=dict(K=K, seed=self.seed, stack=self.stack))
