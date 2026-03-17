import numpy as np
import pandas as pd
import logging

try:
    import keras.ops as _keras_ops
    def _softplus(x):
        return _keras_ops.softplus(x)
    def _log(x):
        return _keras_ops.log(x)
    def _exp(x):
        return _keras_ops.exp(x)
except ImportError:
    def _softplus(x):
        x = np.asarray(x, dtype=np.float32)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    def _log(x):
        return np.log(np.asarray(x, dtype=np.float32))
    def _exp(x):
        return np.exp(np.asarray(x, dtype=np.float32))


class _Identity:
    """Backend-agnostic identity bijector (replaces tfp.bijectors.Identity)."""
    def __init__(self, name=None):
        self.name = name

    def forward(self, x):
        return x

    def inverse(self, y):
        return y


class _Softplus:
    """Backend-agnostic softplus bijector (replaces tfp.bijectors.Softplus)."""
    def __init__(self, name=None):
        self.name = name

    def forward(self, x):
        return _softplus(x)

    def inverse(self, y):
        return _log(_exp(y) - 1.0)


class _Periodic:
    """Backend-agnostic periodic (wrapping) bijector (replaces TFP Periodic)."""
    def __init__(self, low, high, name=None):
        self.low = low
        self.high = high
        self.width = high - low
        self.name = name

    def forward(self, x):
        return ((x - self.low) % self.width) + self.low

    def inverse(self, y):
        return y

class Stimulus(object):

    dimension_labels = ['x']

    def __init__(self, n_dimensions=1):
        if n_dimensions != 1:
            self.dimension_labels = [f'dim_{ix}' for ix in range(n_dimensions)]

        self.bijectors = [_Identity(name=label) for label in self.dimension_labels]

    def clean_paradigm(self, paradigm):

        if (not isinstance(paradigm, pd.DataFrame)) and (paradigm is not None):

            if isinstance(paradigm, pd.Series):
                paradigm = paradigm.to_frame()
                paradigm.columns = self.dimension_labels
            else:
                if paradigm.ndim == 1:
                    paradigm = paradigm[:, np.newaxis]

                paradigm = pd.DataFrame(paradigm, columns=pd.Index(self.dimension_labels, name='stimulus dimensions'),
                index=pd.Index(np.arange(len(paradigm)), name='frame')).astype(np.float32)

        if isinstance(paradigm, pd.DataFrame):
            paradigm = paradigm.astype(np.float32)

        return paradigm

    def _clean_paradigm(self, paradigm):
        if paradigm is not None:
            if isinstance(paradigm, pd.DataFrame):
                return paradigm.values.astype(np.float32)
            elif isinstance(paradigm, np.ndarray):
                return paradigm.astype(np.float32)

            return paradigm

    def _generate_stimulus(self, paradigm):
        return paradigm

    def generate_stimulus(self, paradigm):
        return self.clean_paradigm(paradigm)

    def generate_empty_stimulus(self, size):
        stimulus = np.ones((size, len(self.dimension_labels)), dtype=np.float32) * 1e-6

        if self.bijectors is not None:

            if isinstance(self.bijectors, list):
                stimulus = np.stack([np.asarray(bijector.forward(stimulus[:, ix])) for ix, bijector in enumerate(self.bijectors)], axis=1)
            else:
                stimulus = np.asarray(self.bijectors.forward(stimulus))

        return stimulus.astype(np.float32)


class OneDimensionalStimulusWithAmplitude(Stimulus):
    dimension_labels = ['x', 'amplitude']

    def __init__(self, positive_only=True):
        super().__init__()

        if positive_only:
            self.bijectors = [_Identity(name='x'), _Softplus(name='amplitude')]

class TwoDimensionalStimulus(Stimulus):
    dimension_labels = ['x', 'y']


class OneDimensionalRadialStimulus(Stimulus):
    dimension_labels = ['x (radians)']

    def __init__(self):
        
        super().__init__()
        self.bijectors = [_Periodic(low=0.0, high=2*np.pi, name='x')]


class OneDimensionalRadialStimulusWithAmplitude(OneDimensionalStimulusWithAmplitude):
    dimension_labels = ['x (radians)', 'amplitude']

    def __init__(self, positive_only=True):
        
        super().__init__()

        if positive_only:
            self.bijectors = [_Periodic(low=0.0, high=2*np.pi, name='x'), _Softplus(name='amplitude')]
        else:
            self.bijectors = [_Periodic(low=0.0, high=2*np.pi, name='x'), _Identity(name='amplitude')]

class OneDimensionalGaussianStimulus(Stimulus):
    dimension_labels = ['x', 'sd']

    def __init__(self, positive_only=True):
        super().__init__()

        self.bijectors = [_Identity(name='x'), _Softplus(name='sd')]


class OneDimensionalGaussianStimulusWithAmplitude(Stimulus):
    dimension_labels = ['x', 'sd', 'amplitude']

    def __init__(self, positive_only=True):
        super().__init__()

        if positive_only:
            self.bijectors = [_Identity(name='x'), _Softplus(name='sd'), _Softplus(name='amplitude')]
        else:
            self.bijectors = [_Identity(name='x'), _Softplus(name='sd'), _Identity(name='amplitude')]


class ImageStimulus(Stimulus):

    def __init__(self, grid_coordinates, positive_only=True):
        self.grid_coordinates = pd.DataFrame(grid_coordinates, columns=['x', 'y'])

        if positive_only:
            self.bijectors = _Softplus(name='intensity')
        else:
            self.bijectors = _Identity(name='intensity')

        self.dimension_labels = pd.MultiIndex.from_frame(self.grid_coordinates)

    def clean_paradigm(self, paradigm):

        if (not isinstance(paradigm, pd.DataFrame)) and (paradigm is not None):

            if paradigm.ndim == 3:
                paradigm = paradigm[:, np.newaxis]

                paradigm = paradigm.reshape((paradigm.shape[0], -1))
            elif paradigm.ndim == 2:
                pass
            else:
                raise ValueError('Paradigm should be 2 or 3 dimensional')

            paradigm = pd.DataFrame(paradigm, columns=self.dimension_labels, index=pd.Index(np.arange(len(paradigm)), name='frame')).astype(np.float32)

        if isinstance(paradigm, pd.DataFrame):
            paradigm = paradigm.astype(np.float32)

        return paradigm

    def generate_empty_stimulus(self, size):
        return np.ones((size, len(self.dimension_labels)), dtype=np.float32) * 1e-6
