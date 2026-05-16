"""Tests for braincoder utility functions (formatting, math, stats)."""
import numpy as np
from keras import ops
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# format_paradigm
# ---------------------------------------------------------------------------

class TestFormatParadigm:

    def test_none_returns_none(self):
        from braincoder.utils.formatting import format_paradigm
        assert format_paradigm(None) is None

    def test_dataframe_passthrough(self):
        from braincoder.utils.formatting import format_paradigm
        df = pd.DataFrame({'a': [1.0, 2.0]})
        result = format_paradigm(df)
        assert result is df

    def test_series_to_frame(self):
        from braincoder.utils.formatting import format_paradigm
        s = pd.Series([1.0, 2.0, 3.0])
        result = format_paradigm(s)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_1d_array_to_column(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.array([1.0, 2.0, 3.0])
        result = format_paradigm(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_2d_array_preserved(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.ones((10, 3))
        result = format_paradigm(arr)
        assert result.shape == (10, 3)

    def test_3d_array_flattened(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.ones((10, 4, 4))
        result = format_paradigm(arr)
        assert result.shape == (10, 16)

    def test_dtype_is_float32(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.ones((5, 2), dtype=np.float64)
        result = format_paradigm(arr)
        assert result.dtypes.iloc[0] == np.float32

# ---------------------------------------------------------------------------
# format_parameters
# ---------------------------------------------------------------------------

class TestFormatParameters:

    def test_none_returns_none(self):
        from braincoder.utils.formatting import format_parameters
        assert format_parameters(None) is None

    def test_dataframe_passthrough_float32(self):
        from braincoder.utils.formatting import format_parameters
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        result = format_parameters(df)
        assert isinstance(result, pd.DataFrame)
        assert result.dtypes.iloc[0] == np.float32

    def test_array_with_labels(self):
        from braincoder.utils.formatting import format_parameters
        arr = np.ones((3, 2), dtype=np.float32)
        result = format_parameters(arr, parameter_labels=['a', 'b'])
        assert list(result.columns) == ['a', 'b']
        assert result.shape == (3, 2)

    def test_array_without_labels_uses_default(self):
        from braincoder.utils.formatting import format_parameters
        arr = np.ones((2, 3), dtype=np.float32)
        result = format_parameters(arr)
        assert list(result.columns) == ['par1', 'par2', 'par3']

# ---------------------------------------------------------------------------
# format_data
# ---------------------------------------------------------------------------

class TestFormatData:

    def test_dataframe_passthrough(self):
        from braincoder.utils.formatting import format_data
        df = pd.DataFrame(np.ones((5, 3)))
        result = format_data(df)
        assert result is df

    def test_array_to_dataframe(self):
        from braincoder.utils.formatting import format_data
        arr = np.ones((10, 4), dtype=np.float32)
        result = format_data(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 4)

    def test_dtype_is_float32(self):
        from braincoder.utils.formatting import format_data
        arr = np.ones((5, 2), dtype=np.float64)
        result = format_data(arr)
        assert result.dtypes.iloc[0] == np.float32

    def test_index_name_is_time(self):
        from braincoder.utils.formatting import format_data
        arr = np.ones((5, 2), dtype=np.float32)
        result = format_data(arr)
        assert result.index.name == 'time'

# ---------------------------------------------------------------------------
# format_weights
# ---------------------------------------------------------------------------

class TestFormatWeights:

    def test_none_returns_none(self):
        from braincoder.utils.formatting import format_weights
        assert format_weights(None) is None

    def test_dataframe_passthrough(self):
        from braincoder.utils.formatting import format_weights
        df = pd.DataFrame(np.ones((3, 4)))
        result = format_weights(df)
        assert result is df

    def test_array_to_dataframe(self):
        from braincoder.utils.formatting import format_weights
        arr = np.ones((3, 4), dtype=np.float32)
        result = format_weights(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)

    def test_index_name_is_population(self):
        from braincoder.utils.formatting import format_weights
        arr = np.ones((3, 4), dtype=np.float32)
        result = format_weights(arr)
        assert result.index.name == 'population'

# ---------------------------------------------------------------------------
# gamma_pdf (math utility in hrf.py)
# ---------------------------------------------------------------------------

class TestGammaPdf:

    def test_output_shape(self):
        from braincoder.hrf import gamma_pdf
        t = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        result = ops.convert_to_numpy(gamma_pdf(t, a=6.0, d=1.0))
        assert result.shape == (3, 1)

    def test_positive_values(self):
        from braincoder.hrf import gamma_pdf
        t = np.linspace(0.1, 20.0, 100, dtype=np.float32)[:, np.newaxis]
        result = ops.convert_to_numpy(gamma_pdf(t, a=6.0, d=1.0))
        assert np.all(result >= 0), "Gamma PDF values should be non-negative"

    def test_peak_near_mode(self):
        """Mode of Gamma(a, d) is (a-1)*d."""
        from braincoder.hrf import gamma_pdf
        a, d = 6.0, 1.0
        t = np.linspace(0.1, 20.0, 1000, dtype=np.float32)[:, np.newaxis]
        result = ops.convert_to_numpy(gamma_pdf(t, a=a, d=d))
        peak_t = t.flatten()[np.argmax(result)]
        expected_mode = (a - 1) * d  # = 5.0
        assert abs(peak_t - expected_mode) < 0.5, \
            f"Peak at {peak_t:.2f}, expected ~{expected_mode:.1f}"


# ---------------------------------------------------------------------------
# R² mixture posterior / p_signal threshold
# ---------------------------------------------------------------------------

class TestR2Posterior:
    """The p_signal>0.5 path is now the default voxel-selection rule
    in fit_gp_prior.py. Locks in: (i) p_signal aligns with mixture
    component (high R² → signal); (ii) threshold sits between the
    two component means; (iii) p_signal is monotone in R²."""

    def _make_fit(self, rng, n=2000, w_signal=0.3,
                  noise_mu=-3.0, signal_mu=0.5,
                  noise_sigma=0.6, signal_sigma=0.5):
        """Sample logit-Gaussian mixture, fit it, return (r2, fit)."""
        from braincoder.utils.stats import _inv_logit, fit_r2_mixture
        n_signal = rng.binomial(n, w_signal)
        z = np.concatenate([
            rng.normal(noise_mu,  noise_sigma, n - n_signal),
            rng.normal(signal_mu, signal_sigma, n_signal),
        ])
        r2 = _inv_logit(z)
        return r2, fit_r2_mixture(r2)

    def test_p_signal_is_monotone_in_r2(self):
        from braincoder.utils.stats import r2_posterior_signal
        rng = np.random.default_rng(0)
        r2, fit = self._make_fit(rng)
        grid = np.linspace(0.01, 0.95, 100)
        p = r2_posterior_signal(grid, fit)
        assert np.all(np.diff(p) >= -1e-9), \
            "p_signal must be monotone non-decreasing in R²"

    def test_p_signal_outside_unit_interval_is_zero(self):
        from braincoder.utils.stats import r2_posterior_signal
        rng = np.random.default_rng(1)
        _, fit = self._make_fit(rng)
        r2 = np.array([-0.1, 0.0, 1.0, 1.5, np.nan, np.inf])
        p = r2_posterior_signal(r2, fit)
        assert np.all(p == 0.0)

    def test_threshold_lies_between_component_means(self):
        from braincoder.utils.stats import (
            _inv_logit, r2_p_signal_threshold)
        rng = np.random.default_rng(2)
        _, fit = self._make_fit(rng)
        t = r2_p_signal_threshold(fit, p=0.5)
        # Threshold should be above the noise mean R² and below the
        # signal mean R² for a non-degenerate mixture.
        assert _inv_logit(fit['noise_mu']) < t < _inv_logit(fit['signal_mu'])

    def test_threshold_consistent_with_posterior(self):
        """Voxels with r² > t* should have p_signal ≥ 0.5, and vice versa."""
        from braincoder.utils.stats import (
            r2_posterior_signal, r2_p_signal_threshold)
        rng = np.random.default_rng(3)
        r2, fit = self._make_fit(rng)
        t = r2_p_signal_threshold(fit, p=0.5)
        p = r2_posterior_signal(r2, fit)
        # Tiny grid-quantization tolerance.
        above = r2 > t + 1e-3
        below = r2 < t - 1e-3
        assert (p[above] >= 0.5 - 1e-2).all()
        assert (p[below] <  0.5 + 1e-2).all()
