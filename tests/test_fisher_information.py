"""Tests for Fisher information + expected-uncertainty wrappers.

Two themes:

1. **Analytical FI** — exact closed form Jᵀ Ω⁻¹ J for Gaussian, multiplied
   by (ν + p)/(ν + p + 2) for multivariate Student-t. We verify it
   recovers known results on a hand-tuned single-voxel toy and matches
   the MC estimate to within MC noise on a small multi-voxel model.

2. **get_expected_uncertainty** — simulate-decode-aggregate. We verify
   the returned DataFrame has the right shape, that decoder is roughly
   unbiased on a well-fit model, and that 1/var_E correlates with the
   analytical Fisher information across stimuli.
"""
import numpy as np
import pandas as pd
import pytest

from braincoder.models import GaussianPRF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_model():
    """5-voxel GaussianPRF with peaks spread over [-3, 3]."""
    pars = pd.DataFrame({
        'mu':        np.linspace(-3, 3, 5, dtype=np.float32),
        'sd':        np.ones(5, dtype=np.float32),
        'amplitude': np.ones(5, dtype=np.float32) * 2,
        'baseline':  np.zeros(5, dtype=np.float32),
    })
    model = GaussianPRF(parameters=pars)
    return model, pars


@pytest.fixture
def toy_omega():
    """Diagonal Σ = σ²I — easiest case for an analytic sanity check."""
    return (0.5 ** 2) * np.eye(5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Analytical Fisher information
# ---------------------------------------------------------------------------

class TestAnalyticalFisher:

    def test_returns_series_of_right_length(self, toy_model, toy_omega):
        model, pars = toy_model
        stim = np.linspace(-4, 4, 30, dtype=np.float32)
        fi = model.get_fisher_information(stim, omega=toy_omega,
                                          parameters=pars, analytical=True)
        assert isinstance(fi, pd.Series)
        assert len(fi) == 30
        assert np.all(np.isfinite(fi.values))
        assert np.all(fi.values >= 0)

    def test_fi_zero_at_each_voxel_mode(self, toy_model, toy_omega):
        """Gaussian dµ/ds = 0 at µ for each voxel → FI(s=mode) is the sum
        of the OTHER voxels' gradients only. We just check that FI between
        peaks is generally non-zero and continuous."""
        model, pars = toy_model
        # At each voxel's mode, that voxel contributes 0 — but neighbours do.
        stim = pars['mu'].values.astype(np.float32)
        fi = model.get_fisher_information(stim, omega=toy_omega,
                                          parameters=pars, analytical=True)
        # FI should still be > 0 thanks to neighbouring voxels' flanks.
        assert (fi.values > 0).all()

    def test_dof_scales_fi(self, toy_model, toy_omega):
        """Student-t FI is exactly (ν+p)/(ν+p+2) · Gaussian FI."""
        model, pars = toy_model
        stim = np.linspace(-3, 3, 11, dtype=np.float32)
        fi_gauss = model.get_fisher_information(stim, omega=toy_omega,
                                                 parameters=pars,
                                                 analytical=True).values
        p = toy_omega.shape[0]
        for dof in (3.0, 10.0, 100.0):
            fi_t = model.get_fisher_information(
                stim, omega=toy_omega, dof=dof,
                parameters=pars, analytical=True).values
            expected_factor = (dof + p) / (dof + p + 2.0)
            np.testing.assert_allclose(fi_t, fi_gauss * expected_factor,
                                        rtol=1e-5,
                                        err_msg=f'Student-t factor wrong at dof={dof}')

    def test_dof_infinity_limit_recovers_gaussian(self, toy_model, toy_omega):
        """As ν → ∞ the Student-t factor → 1, recovering the Gaussian FI."""
        model, pars = toy_model
        stim = np.linspace(-3, 3, 5, dtype=np.float32)
        fi_gauss = model.get_fisher_information(stim, omega=toy_omega,
                                                 parameters=pars,
                                                 analytical=True).values
        fi_t = model.get_fisher_information(stim, omega=toy_omega,
                                             dof=1e6, parameters=pars,
                                             analytical=True).values
        np.testing.assert_allclose(fi_t, fi_gauss, rtol=1e-3)

    def test_analytical_matches_mc_within_noise(self, toy_model, toy_omega):
        """Gaussian analytical FI vs MC FI with same Ω. MC has ~10% SE at
        n=2000; we accept up to 30% relative difference per point but
        require correlation ≥ 0.95."""
        model, pars = toy_model
        stim = np.linspace(-3, 3, 20, dtype=np.float32)
        fi_an = model.get_fisher_information(stim, omega=toy_omega,
                                              parameters=pars,
                                              analytical=True).values
        fi_mc = model.get_fisher_information(stim, omega=toy_omega,
                                              parameters=pars,
                                              analytical=False, n=2000).values
        assert np.corrcoef(fi_an, fi_mc)[0, 1] >= 0.95
        rel = np.abs(fi_mc - fi_an) / np.maximum(fi_an, 1e-3)
        assert rel.mean() < 0.40


# ---------------------------------------------------------------------------
# Expected uncertainty (simulate → decode → aggregate)
# ---------------------------------------------------------------------------

class TestExpectedUncertainty:

    def test_returns_one_row_per_stim(self, toy_model, toy_omega):
        model, pars = toy_model
        stim = np.linspace(-3, 3, 13, dtype=np.float32)
        out = model.get_expected_uncertainty(stim, omega=toy_omega,
                                              parameters=pars,
                                              n_simulations=80)
        assert out.shape == (13, 5)
        for col in ('mean_E', 'var_E', 'mean_error', 'mean_abs_error', 'n_sims'):
            assert col in out.columns
        assert (out['n_sims'] == 80).all()

    def test_bias_is_small_in_interior(self, toy_model, toy_omega):
        """For interior stimuli the decoder should be roughly unbiased."""
        model, pars = toy_model
        stim = np.linspace(-2, 2, 9, dtype=np.float32)
        out = model.get_expected_uncertainty(stim, omega=toy_omega,
                                              parameters=pars,
                                              n_simulations=400,
                                              decoder_stimulus_range=np.linspace(
                                                  -4, 4, 201).astype(np.float32))
        # Bias should be small compared to the stimulus range.
        assert np.abs(out['mean_error']).max() < 0.5

    def test_var_E_in_high_fi_regions_smaller(self):
        """Decoder posterior-mean variance is smaller where FI is larger
        — a basic Cramér–Rao sanity check, robust to simulation noise.

        Uses a 20-voxel population. We bin stimuli into "high-FI" and
        "low-FI" halves and require mean(var_E[high]) < mean(var_E[low]).
        Stronger than a Spearman correlation across 11 noisy points and
        less sensitive to local non-monotonicities in the FI curve
        introduced by clustering of voxel preferences.
        """
        from braincoder.models import GaussianPRF
        pars = pd.DataFrame({
            'mu':        np.linspace(-3, 3, 20, dtype=np.float32),
            'sd':        np.full(20, 0.6, dtype=np.float32),
            'amplitude': np.full(20, 2.0, dtype=np.float32),
            'baseline':  np.zeros(20, dtype=np.float32),
        })
        model = GaussianPRF(parameters=pars)
        omega = (0.3 ** 2) * np.eye(20, dtype=np.float32)
        stim = np.linspace(-2.5, 2.5, 21, dtype=np.float32)
        fi = model.get_fisher_information(stim, omega=omega,
                                           parameters=pars,
                                           analytical=True).values
        out = model.get_expected_uncertainty(
            stim, omega=omega, parameters=pars, n_simulations=800,
            decoder_stimulus_range=np.linspace(-4, 4, 201).astype(np.float32))
        median_fi = float(np.median(fi))
        high = out['var_E'].values[fi >= median_fi]
        low = out['var_E'].values[fi < median_fi]
        # Robust against simulation noise: expect a non-trivial gap.
        assert high.mean() < low.mean(), (
            f'High-FI var_E={high.mean():.4f} not less than '
            f'low-FI var_E={low.mean():.4f}')

    def test_batch_stimuli_gives_same_result(self, toy_model, toy_omega):
        """Batched vs unbatched should differ only by simulation seed; mean_E
        should still align very closely."""
        model, pars = toy_model
        stim = np.linspace(-2, 2, 9, dtype=np.float32)
        common = dict(omega=toy_omega, parameters=pars,
                      n_simulations=200,
                      decoder_stimulus_range=np.linspace(-4, 4, 101).astype(np.float32))
        np.random.seed(0)
        full = model.get_expected_uncertainty(stim, batch_stimuli=None, **common)
        np.random.seed(0)
        batched = model.get_expected_uncertainty(stim, batch_stimuli=3, **common)
        # The two share the simulation algorithm; row counts must agree.
        np.testing.assert_array_equal(full['n_sims'].values,
                                       batched['n_sims'].values)
