"""Tests for braincoder.selection (label-permutation voxel selection).

Most tests use a cheap numpy fitting procedure (best-of-grid Gaussian tuning
curve, R² = max squared correlation over the grid), because FDR calibration
needs hundreds of simulated datasets. The p-value / BH / shuffling machinery
does not care what the fitting callable is. One test runs the real braincoder
path (``make_r2_fitter`` + ``ParameterFitter``).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from braincoder.selection import (PermutationSelector, PermutationSelectionResult,
                                  bh_qvalues, permutation_indices, pooled_pvalues,
                                  voxelwise_pvalues)

MUS = np.linspace(0, 10, 11)
SDS = np.array([1., 2., 4.])


def grid_r2(data, paradigm):
    """R² of the best Gaussian tuning curve on a (mu, sd) grid, per voxel.

    A flexible procedure that overfits noise (max over 33 candidate curves),
    like a real grid search does. Row-wise: row i's prediction depends only on
    stimulus i, so ``stack=True`` is valid.
    """
    x = np.asarray(paradigm, dtype=float).ravel()
    y = np.asarray(data, dtype=float)
    y = (y - y.mean(0)) / y.std(0)
    best = np.zeros(y.shape[1])
    for mu in MUS:
        for sd in SDS:
            b = np.exp(-0.5 * ((x - mu) / sd) ** 2)
            b = (b - b.mean()) / b.std()
            best = np.maximum(best, (b @ y / len(x)) ** 2)
    return best


def simulate(n_runs=8, n_per_run=20, n_vox=300, n_signal=0, amplitude=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_runs * n_per_run
    runs = np.repeat(np.arange(n_runs), n_per_run)
    x = rng.uniform(0, 10, n)
    y = rng.normal(size=(n, n_vox))
    mus = rng.uniform(2, 8, n_signal)
    for v in range(n_signal):
        y[:, v] += amplitude * np.exp(-0.5 * ((x - mus[v]) / 1.5) ** 2)
    idx = pd.MultiIndex.from_arrays([runs, np.arange(n)], names=['run', 'trial'])
    return (pd.DataFrame(y, index=idx),
            pd.DataFrame({'x': x}, index=idx), runs)


# ---------------------------------------------------------------------------
# shuffles

def test_shuffles_respect_groups():
    groups = np.repeat(['a', 'b', 'c', 'd'], [5, 7, 3, 9])
    perms = permutation_indices(len(groups), groups, n_permutations=20, seed=3)
    assert perms.shape == (20, len(groups))
    for pm in perms:
        assert np.array_equal(np.sort(pm), np.arange(len(groups)))
        assert np.array_equal(groups[pm], groups)
    # not the identity (with overwhelming probability)
    assert (perms != np.arange(len(groups))).any(1).all()


def test_groups_from_index_levels():
    data, paradigm, runs = simulate(n_runs=4, n_per_run=10, n_vox=5)
    sel = PermutationSelector(grid_r2, n_permutations=4, groups='run', seed=1)
    res = sel.fit(data, paradigm)
    for pm in res.permutations:
        assert np.array_equal(runs[pm], runs)


def test_seeds_reproducible_and_extendable():
    g = np.repeat(np.arange(5), 10)
    a = permutation_indices(50, g, 5, seed=(6, 1, 3))
    b = permutation_indices(50, g, 5, seed=(6, 1, 3))
    c = permutation_indices(50, g, 5, seed=(6, 1, 4))
    d = permutation_indices(50, g, 8, seed=(6, 1, 3))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.array_equal(a, d[:5])           # more permutations extend the set
    with pytest.raises(ValueError):
        permutation_indices(50, g, 5, seed=None)

    data, paradigm, runs = simulate(n_vox=20, seed=2)
    r1 = PermutationSelector(grid_r2, 3, groups=runs, seed=7).fit(data, paradigm)
    r2 = PermutationSelector(grid_r2, 3, groups=runs, seed=7).fit(data, paradigm)
    pd.testing.assert_frame_equal(r1.null_r2, r2.null_r2)
    pd.testing.assert_series_equal(r1.q, r2.q)


# ---------------------------------------------------------------------------
# p-values

def test_pvalue_formulas():
    null = np.array([[0.1, 0.2], [0.3, 0.4]])
    p = pooled_pvalues(np.array([0.25, 0.5, np.nan]), null)
    np.testing.assert_allclose(p[:2], [(1 + 2) / 5, 1 / 5])
    assert np.isnan(p[2])
    pv = voxelwise_pvalues(np.array([0.2, 0.0]), null)
    np.testing.assert_allclose(pv, [(1 + 1) / 3, (1 + 2) / 3])
    q = bh_qvalues(np.array([0.01, 0.04, 0.03, np.nan]))
    np.testing.assert_allclose(q[:3], [0.03, 0.04, 0.04])
    assert np.isnan(q[3])


def test_stack_equals_loop():
    data, paradigm, runs = simulate(n_vox=40, n_signal=5, seed=4)
    a = PermutationSelector(grid_r2, 4, groups=runs, seed=0, stack=False).fit(data, paradigm)
    b = PermutationSelector(grid_r2, 4, groups=runs, seed=0, stack=True).fit(data, paradigm)
    np.testing.assert_allclose(a.r2.values, b.r2.values, rtol=1e-10)
    np.testing.assert_allclose(a.null_r2.values, b.null_r2.values, rtol=1e-10)
    pd.testing.assert_series_equal(a.mask, b.mask)


# ---------------------------------------------------------------------------
# FDR control and power

@pytest.mark.parametrize('pool_within_family', [False, True])
def test_fdr_controlled_on_pure_noise(pool_within_family):
    """Complete null: BH controls P(any discovery) at q. 200 datasets."""
    n_any = 0
    n_sim = 200
    fam = np.repeat(['L', 'R'], 150)
    for s in range(n_sim):
        data, paradigm, runs = simulate(n_vox=300, seed=1000 + s)
        res = PermutationSelector(grid_r2, 5, groups=runs, q=0.05, seed=s,
                                  stack=True).fit(
            data, paradigm, bh_groups=fam, pool_within_family=pool_within_family)
        # one family per ROI: count a dataset as a false positive if either has one
        n_any += res.mask.any()
    rate = n_any / n_sim
    # two families at q = .05 each -> nominal family-wise rate <= ~.10;
    # check each family separately below for the per-family bound.
    assert rate <= 0.14, rate


@pytest.mark.parametrize('K', [5, 50])
def test_fdr_per_family_rate(K):
    """P(any discovery) <= q under the complete null. With K = 5 the pool's
    granularity makes this very conservative (BH needs ~1/(qK) = 4 voxels
    above every null value); K = 50 tests the rule near its nominal level
    (measured 0.045 over 400 datasets)."""
    n_sim, hits = 200, 0
    for s in range(n_sim):
        data, paradigm, runs = simulate(n_vox=300, seed=5000 + s)
        res = PermutationSelector(grid_r2, K, groups=runs, q=0.05, seed=s,
                                  stack=True).fit(data, paradigm)
        hits += res.mask.any()
    assert hits / n_sim <= 0.09, hits / n_sim   # nominal .05 + MC error (SE .015)


def test_power_on_signal_voxels():
    data, paradigm, runs = simulate(n_vox=400, n_signal=40, amplitude=1.2, seed=11)
    res = PermutationSelector(grid_r2, 5, groups=runs, q=0.05, seed=0,
                              stack=True).fit(data, paradigm)
    sig = res.mask.values[:40]
    noise = res.mask.values[40:]
    assert sig.mean() >= 0.9, sig.mean()
    # false discovery proportion among selections stays near q
    fdp = noise.sum() / max(res.n_selected, 1)
    assert fdp <= 0.15, fdp
    # unpooled p cannot go below 1 / (K + 1)
    assert res.p_voxelwise.min() >= 1 / 6 - 1e-12
    cal = res.calibration()
    assert set(cal) >= {'any_discovery_rate', 'n_discoveries'}


def test_bh_families_and_untested():
    data, paradigm, runs = simulate(n_vox=60, n_signal=10, amplitude=1.5, seed=5)
    fam = np.array(['A'] * 30 + ['B'] * 20 + [None] * 10, dtype=object)
    res = PermutationSelector(grid_r2, 5, groups=runs, seed=1).fit(
        data, paradigm, bh_groups=fam)
    assert res.q.iloc[50:].isna().all()
    assert not res.mask.iloc[50:].any()
    summ = res.summary()
    assert list(summ.family) == ['A', 'B']
    assert summ.n_tested.tolist() == [30, 20]
    # BH within family == BH on that family's p alone
    np.testing.assert_allclose(res.q.values[30:50], bh_qvalues(res.p.values[30:50]))


# ---------------------------------------------------------------------------
# exchangeability diagnostic

def _fake_result(null, r2=None):
    K, V = null.shape
    cols = np.arange(V)
    r2 = np.zeros(V) if r2 is None else r2
    return PermutationSelectionResult(
        r2=pd.Series(r2, index=cols), null_r2=pd.DataFrame(null, columns=cols),
        p=pd.Series(np.nan, index=cols), q=pd.Series(np.nan, index=cols),
        mask=pd.Series(False, index=cols), q_thresh=.05, method='pooled',
        permutations=np.zeros((K, 1), int))


def test_exchangeability_report():
    rng = np.random.default_rng(0)
    ok = _fake_result(rng.gamma(2, 0.01, size=(10, 2000)))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rep = ok.exchangeability_report()
    assert rep['exchangeable'] is True
    assert abs(rep['split_half_r']) < 0.1

    # voxel-specific null: half the voxels have a 3x larger null scale
    scale = np.where(np.arange(2000) < 1000, 0.01, 0.03)
    bad = _fake_result(rng.gamma(2, 1, size=(10, 2000)) * scale)
    with pytest.warns(RuntimeWarning, match='exchangeable'):
        rep = bad.exchangeability_report()
    assert rep['exchangeable'] is False
    assert rep['split_half_r'] > 0.3


def test_real_braincoder_fit():
    """End-to-end with a braincoder model and make_r2_fitter."""
    from braincoder.models import GaussianPRF

    rng = np.random.default_rng(1)
    n_runs, n_per = 6, 20
    n = n_runs * n_per
    runs = np.repeat(np.arange(n_runs), n_per)
    x = rng.uniform(0, 10, n).astype(np.float32)
    paradigm = pd.DataFrame({'x': x})
    model = GaussianPRF()
    n_sig, n_vox = 8, 24
    pars = pd.DataFrame({'mu': rng.uniform(3, 7, n_sig), 'sd': 1.5,
                         'amplitude': 1.0, 'baseline': 0.0}).astype(np.float32)
    sig = model.predict(parameters=pars, paradigm=paradigm).values
    y = rng.normal(0, 0.5, (n, n_vox)).astype(np.float32)
    y[:, :n_sig] += sig
    data = pd.DataFrame(y)

    grid = (np.linspace(0, 10, 11).astype(np.float32),
            np.array([1., 2., 4.], np.float32), [1.], [0.])
    sel = PermutationSelector(model, n_permutations=4, groups=runs, q=0.05,
                              seed=(1, 2), stack=True)
    res = sel.fit(data, paradigm, fit_kwargs=dict(grid=grid, max_n_iterations=150,
                                                   learning_rate=0.05))
    assert res.null_r2.shape == (4, n_vox)
    assert res.mask.values[:n_sig].mean() >= 0.75
    assert res.mask.values[n_sig:].sum() <= 1


# ---------------------------------------------------------------------------
# cached null (NullBank)

from braincoder.selection import NullBank, subsample_voxels  # noqa: E402


def test_null_bank_fdr_on_pure_noise(tmp_path):
    """Null fitted on a 300-voxel subsample, saved, reloaded, and used to
    select among all 600 voxels (real R² from a separate fit): P(any
    discovery) stays <= q."""
    n_sim, hits = 200, 0
    for s in range(n_sim):
        data, paradigm, runs = simulate(n_vox=600, seed=9000 + s)
        bank = PermutationSelector(grid_r2, 20, groups=runs, seed=s, stack=True).fit_null(
            data, paradigm, n_voxels=300, voxel_seed=s, metadata=dict(model='grid'))
        if s == 0:
            fn = tmp_path / 'bank.npz'
            bank.save(fn)
            b2 = NullBank.load(fn)
            np.testing.assert_allclose(b2.null_r2, bank.null_r2, rtol=1e-6)
            assert b2.metadata['model'] == 'grid' and b2.K == 20
            np.testing.assert_array_equal(b2.voxel_ids, bank.voxel_ids)
        hits += bank.select(grid_r2(data, paradigm), q=0.05).mask.any()
    assert hits / n_sim <= 0.09, hits / n_sim


def test_subsample_null_matches_full_null():
    """Exchangeable data: p against a subsample null ~= p against the null of
    all voxels; the selections agree."""
    data, paradigm, runs = simulate(n_vox=2000, n_signal=100, amplitude=1.0, seed=21)
    sel = PermutationSelector(grid_r2, 10, groups=runs, seed=3, stack=True)
    full = sel.fit_null(data, paradigm)
    sub = sel.fit_null(data, paradigm, n_voxels=500, voxel_seed=1)
    r2 = grid_r2(data, paradigm)
    pf, ps = full.pvalues(r2), sub.pvalues(r2)
    ok = pf < 0.5
    assert np.corrcoef(np.log(pf[ok]), np.log(ps[ok]))[0, 1] > 0.98
    mf, ms = full.select(r2).mask, sub.select(r2).mask
    assert (mf == ms).mean() > 0.98
    assert mf[:100].mean() > 0.8


def test_null_bank_calibration_and_subsample():
    ids = subsample_voxels(1000, 200, seed=(4, 2))
    assert len(ids) == 200 and np.all(np.diff(ids) > 0)
    np.testing.assert_array_equal(ids, subsample_voxels(1000, 200, seed=(4, 2)))
    np.testing.assert_array_equal(subsample_voxels(50, 200, 0), np.arange(50))
    data, paradigm, runs = simulate(n_vox=400, seed=8)
    bank = PermutationSelector(grid_r2, 10, groups=runs, seed=0, stack=True).fit_null(data, paradigm)
    cal = bank.calibration(bank.null_r2)
    assert cal['any_discovery_rate'] <= 0.2
    assert len(cal['n_discoveries']) == 10
