"""Tests for braincoder.stimuli — bijector classes and Stimulus subclasses."""
import numpy as np
import pandas as pd
import pytest

from braincoder.stimuli import (
    _Identity, _Softplus, _Periodic,
    Stimulus, TwoDimensionalStimulus,
    OneDimensionalStimulusWithAmplitude,
    OneDimensionalRadialStimulus,
    OneDimensionalRadialStimulusWithAmplitude,
    OneDimensionalGaussianStimulus,
    OneDimensionalGaussianStimulusWithAmplitude,
    ImageStimulus,
)


# ---------------------------------------------------------------------------
# Bijector: _Identity
# ---------------------------------------------------------------------------

class TestIdentityBijector:

    def test_forward_is_identity(self):
        b = _Identity(name='x')
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        np.testing.assert_array_equal(b.forward(x), x)

    def test_inverse_is_identity(self):
        b = _Identity()
        y = np.array([2.5, -3.0], dtype=np.float32)
        np.testing.assert_array_equal(b.inverse(y), y)

    def test_name_stored(self):
        b = _Identity(name='amplitude')
        assert b.name == 'amplitude'

    def test_roundtrip(self):
        b = _Identity()
        x = np.linspace(-10, 10, 50, dtype=np.float32)
        np.testing.assert_array_equal(b.inverse(b.forward(x)), x)


# ---------------------------------------------------------------------------
# Bijector: _Softplus
# ---------------------------------------------------------------------------

class TestSoftplusBijector:

    def test_forward_positive(self):
        b = _Softplus()
        x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        result = np.asarray(b.forward(x))
        assert np.all(result > 0), "Softplus output must be positive"

    def test_forward_near_log2_at_zero(self):
        b = _Softplus()
        result = float(np.asarray(b.forward(np.array([0.0], dtype=np.float32)))[0])
        assert abs(result - np.log(2.0)) < 1e-4

    def test_forward_large_x_approx_x(self):
        b = _Softplus()
        x = np.array([20.0], dtype=np.float32)
        result = float(np.asarray(b.forward(x))[0])
        assert abs(result - 20.0) < 1e-3

    def test_inverse_roundtrip(self):
        b = _Softplus()
        y = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        reconstructed = np.asarray(b.inverse(np.asarray(b.forward(y))))
        np.testing.assert_allclose(reconstructed, y, atol=1e-4)

    def test_name_stored(self):
        b = _Softplus(name='sd')
        assert b.name == 'sd'


# ---------------------------------------------------------------------------
# Bijector: _Periodic
# ---------------------------------------------------------------------------

class TestPeriodicBijector:

    def test_values_in_range_unchanged(self):
        b = _Periodic(low=0.0, high=2 * np.pi)
        x = np.array([0.5, 1.0, np.pi], dtype=np.float32)
        result = np.asarray(b.forward(x))
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_wraps_above_high(self):
        b = _Periodic(low=0.0, high=2 * np.pi)
        x = np.array([2 * np.pi + 0.5], dtype=np.float32)
        result = float(np.asarray(b.forward(x))[0])
        assert abs(result - 0.5) < 1e-5

    def test_wraps_below_low(self):
        b = _Periodic(low=0.0, high=2 * np.pi)
        x = np.array([-0.5], dtype=np.float32)
        result = float(np.asarray(b.forward(x))[0])
        assert abs(result - (2 * np.pi - 0.5)) < 1e-5

    def test_inverse_is_identity(self):
        b = _Periodic(low=0.0, high=2 * np.pi)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(b.inverse(y), y)

    def test_width_stored(self):
        b = _Periodic(low=1.0, high=5.0)
        assert b.width == 4.0

    def test_name_stored(self):
        b = _Periodic(low=0.0, high=1.0, name='phase')
        assert b.name == 'phase'


# ---------------------------------------------------------------------------
# Stimulus (base class)
# ---------------------------------------------------------------------------

class TestStimulus:

    def test_default_dimension_label(self):
        s = Stimulus()
        assert s.dimension_labels == ['x']

    def test_multidimensional_labels(self):
        s = Stimulus(n_dimensions=3)
        assert s.dimension_labels == ['dim_0', 'dim_1', 'dim_2']

    def test_default_bijectors_are_identity(self):
        s = Stimulus()
        assert len(s.bijectors) == 1
        assert isinstance(s.bijectors[0], _Identity)

    def test_clean_paradigm_none(self):
        s = Stimulus()
        assert s.clean_paradigm(None) is None

    def test_clean_paradigm_dataframe_float32(self):
        s = Stimulus()
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        result = s.clean_paradigm(df)
        assert result.dtypes.iloc[0] == np.float32

    def test_clean_paradigm_series_to_frame(self):
        s = Stimulus()
        ser = pd.Series([1.0, 2.0, 3.0])
        result = s.clean_paradigm(ser)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['x']

    def test_clean_paradigm_1d_array(self):
        s = Stimulus()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = s.clean_paradigm(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_clean_paradigm_2d_array(self):
        s = Stimulus()
        arr = np.ones((10, 1), dtype=np.float32)
        result = s.clean_paradigm(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_generate_empty_stimulus_shape(self):
        s = Stimulus()
        out = s.generate_empty_stimulus(5)
        assert out.shape == (5, 1)
        assert out.dtype == np.float32

    def test_generate_empty_stimulus_positive(self):
        s = Stimulus()
        out = s.generate_empty_stimulus(3)
        assert np.all(out > 0)

    def test_generate_stimulus_returns_clean_paradigm(self):
        s = Stimulus()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = s.generate_stimulus(arr)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# TwoDimensionalStimulus
# ---------------------------------------------------------------------------

class TestTwoDimensionalStimulus:

    def test_dimension_labels(self):
        s = TwoDimensionalStimulus()
        assert s.dimension_labels == ['x', 'y']

    def test_generate_empty_stimulus_shape(self):
        s = TwoDimensionalStimulus()
        out = s.generate_empty_stimulus(4)
        assert out.shape == (4, 2)


# ---------------------------------------------------------------------------
# OneDimensionalStimulusWithAmplitude
# ---------------------------------------------------------------------------

class TestOneDimensionalStimulusWithAmplitude:

    def test_dimension_labels(self):
        s = OneDimensionalStimulusWithAmplitude()
        assert s.dimension_labels == ['x', 'amplitude']

    def test_positive_only_bijectors(self):
        s = OneDimensionalStimulusWithAmplitude(positive_only=True)
        assert isinstance(s.bijectors[0], _Identity)
        assert isinstance(s.bijectors[1], _Softplus)

    def test_not_positive_only_uses_identity_for_amplitude(self):
        s = OneDimensionalStimulusWithAmplitude(positive_only=False)
        assert isinstance(s.bijectors[0], _Identity)
        assert isinstance(s.bijectors[1], _Identity)

    def test_generate_empty_stimulus_shape(self):
        s = OneDimensionalStimulusWithAmplitude()
        out = s.generate_empty_stimulus(3)
        assert out.shape == (3, 2)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# OneDimensionalRadialStimulus
# ---------------------------------------------------------------------------

class TestOneDimensionalRadialStimulus:

    def test_bijector_is_periodic(self):
        s = OneDimensionalRadialStimulus()
        assert len(s.bijectors) == 1
        assert isinstance(s.bijectors[0], _Periodic)

    def test_periodic_range(self):
        s = OneDimensionalRadialStimulus()
        b = s.bijectors[0]
        assert b.low == 0.0
        assert abs(b.high - 2 * np.pi) < 1e-6

    def test_generate_empty_stimulus_in_range(self):
        s = OneDimensionalRadialStimulus()
        out = s.generate_empty_stimulus(10)
        assert np.all(out >= 0.0)
        assert np.all(out < 2 * np.pi)


# ---------------------------------------------------------------------------
# OneDimensionalRadialStimulusWithAmplitude
# ---------------------------------------------------------------------------

class TestOneDimensionalRadialStimulusWithAmplitude:

    def test_periodic_and_softplus_bijectors(self):
        s = OneDimensionalRadialStimulusWithAmplitude(positive_only=True)
        assert isinstance(s.bijectors[0], _Periodic)
        assert isinstance(s.bijectors[1], _Softplus)

    def test_not_positive_only(self):
        s = OneDimensionalRadialStimulusWithAmplitude(positive_only=False)
        assert isinstance(s.bijectors[0], _Periodic)
        assert isinstance(s.bijectors[1], _Identity)

    def test_generate_empty_stimulus_shape(self):
        s = OneDimensionalRadialStimulusWithAmplitude()
        out = s.generate_empty_stimulus(5)
        assert out.shape == (5, 2)


# ---------------------------------------------------------------------------
# OneDimensionalGaussianStimulus
# ---------------------------------------------------------------------------

class TestOneDimensionalGaussianStimulus:

    def test_bijectors(self):
        s = OneDimensionalGaussianStimulus()
        assert isinstance(s.bijectors[0], _Identity)
        assert isinstance(s.bijectors[1], _Softplus)

    def test_generate_empty_stimulus_sd_positive(self):
        s = OneDimensionalGaussianStimulus()
        out = s.generate_empty_stimulus(4)
        assert out.shape == (4, 2)
        assert np.all(out[:, 1] > 0), "sd dimension should be positive after softplus"


# ---------------------------------------------------------------------------
# OneDimensionalGaussianStimulusWithAmplitude
# ---------------------------------------------------------------------------

class TestOneDimensionalGaussianStimulusWithAmplitude:

    def test_bijectors_positive_only(self):
        s = OneDimensionalGaussianStimulusWithAmplitude(positive_only=True)
        assert isinstance(s.bijectors[0], _Identity)
        assert isinstance(s.bijectors[1], _Softplus)
        assert isinstance(s.bijectors[2], _Softplus)

    def test_bijectors_not_positive_only(self):
        s = OneDimensionalGaussianStimulusWithAmplitude(positive_only=False)
        assert isinstance(s.bijectors[2], _Identity)

    def test_generate_empty_stimulus_shape(self):
        s = OneDimensionalGaussianStimulusWithAmplitude()
        out = s.generate_empty_stimulus(3)
        assert out.shape == (3, 3)


# ---------------------------------------------------------------------------
# ImageStimulus
# ---------------------------------------------------------------------------

class TestImageStimulus:

    @pytest.fixture
    def grid(self):
        xs = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        ys = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        return pd.DataFrame({'x': xs, 'y': ys})

    def test_bijector_is_softplus(self, grid):
        s = ImageStimulus(grid, positive_only=True)
        assert isinstance(s.bijectors, _Softplus)

    def test_bijector_not_positive_is_identity(self, grid):
        s = ImageStimulus(grid, positive_only=False)
        assert isinstance(s.bijectors, _Identity)

    def test_dimension_labels_is_multiindex(self, grid):
        s = ImageStimulus(grid)
        assert isinstance(s.dimension_labels, pd.MultiIndex)
        assert len(s.dimension_labels) == 4

    def test_clean_paradigm_2d_array(self, grid):
        s = ImageStimulus(grid)
        arr = np.ones((5, 4), dtype=np.float32)
        result = s.clean_paradigm(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 4)

    def test_clean_paradigm_3d_array(self, grid):
        s = ImageStimulus(grid)
        arr = np.ones((5, 2, 2), dtype=np.float32)
        result = s.clean_paradigm(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 4)

    def test_generate_empty_stimulus_shape(self, grid):
        s = ImageStimulus(grid)
        out = s.generate_empty_stimulus(6)
        assert out.shape == (6, 4)
        assert out.dtype == np.float32
