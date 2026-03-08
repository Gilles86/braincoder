"""Tests for braincoder utility functions (formatting, math, stats)."""
import numpy as np
import pandas as pd
import pytest
from braincoder.utils.backend import to_numpy


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
        result = to_numpy(gamma_pdf(t, a=6.0, d=1.0))
        assert result.shape == (3, 1)

    def test_positive_values(self):
        from braincoder.hrf import gamma_pdf
        t = np.linspace(0.1, 20.0, 100, dtype=np.float32)[:, np.newaxis]
        result = to_numpy(gamma_pdf(t, a=6.0, d=1.0))
        assert np.all(result >= 0), "Gamma PDF values should be non-negative"

    def test_peak_near_mode(self):
        """Mode of Gamma(a, d) is (a-1)*d."""
        from braincoder.hrf import gamma_pdf
        a, d = 6.0, 1.0
        t = np.linspace(0.1, 20.0, 1000, dtype=np.float32)[:, np.newaxis]
        result = to_numpy(gamma_pdf(t, a=a, d=d))
        peak_t = t.flatten()[np.argmax(result)]
        expected_mode = (a - 1) * d  # = 5.0
        assert abs(peak_t - expected_mode) < 0.5, \
            f"Peak at {peak_t:.2f}, expected ~{expected_mode:.1f}"
