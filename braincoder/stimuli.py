import tensorflow_probability as tfp
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from .utils.mcmc import cleanup_chain, sample_hmc, Periodic

class Stimulus(object):

    dimension_labels = ['x']

    def __init__(self, n_dimensions=1):
        if n_dimensions != 1:
            self.dimension_labels = [f'dim_{ix}' for ix in range(n_dimensions)]

        self.bijectors = [tfp.bijectors.Identity(name=label) for label in self.dimension_labels]

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
                stimulus = np.stack([bijector.forward(stimulus[:, ix]).numpy() for ix, bijector in enumerate(self.bijectors)], axis=1)
            else:
                stimulus = self.bijectors.forward(stimulus).numpy()

        return stimulus.astype(np.float32)


class OneDimensionalStimulusWithAmplitude(Stimulus):
    dimension_labels = ['x', 'amplitude']

    def __init__(self, positive_only=True):
        
        return super().__init__()

        if positive_only:
            self.bijectors = [tfp.bijectors.Identity(name='x'), tfp.bijectors.Softplus(name='amplitude')]

class TwoDimensionalStimulus(Stimulus):
    dimension_labels = ['x', 'y']


class OneDimensionalRadialStimulus(Stimulus):
    dimension_labels = ['x (radians)']

    def __init__(self):
        
        super().__init__()
        self.bijectors = [Periodic(low=0.0, high=2*np.pi, name='x')]


class OneDimensionalRadialStimulusWithAmplitude(OneDimensionalStimulusWithAmplitude):
    dimension_labels = ['x (radians)', 'amplitude']

    def __init__(self, positive_only=True):
        
        super().__init__()

        if positive_only:
            self.bijectors = [Periodic(low=0.0, high=2*np.pi, name='x'), tfp.bijectors.Softplus(name='amplitude')]
        else:
            self.bijectors = [Periodic(low=0.0, high=2*np.pi, name='x'), tfp.bijectors.Identity(name='amplitude')]

class OneDimensionalGaussianStimulus(Stimulus):
    dimension_labels = ['x', 'sd']

    def __init__(self, positive_only=True):
        super().__init__()

        self.bijectors = [tfp.bijectors.Identity(name='x'), tfp.bijectors.Softplus(name='sd')]


class OneDimensionalGaussianStimulusWithAmplitude(Stimulus):
    dimension_labels = ['x', 'sd', 'amplitude']

    def __init__(self, positive_only=True):
        super().__init__()

        if positive_only:
            self.bijectors = [tfp.bijectors.Identity(name='x'), tfp.bijectors.Softplus(name='sd'), tfp.bijectors.Softplus(name='amplitude')]
        else:
            self.bijectors = [tfp.bijectors.Identity(name='x'), tfp.bijectors.Softplus(name='sd'), tfp.bijectors.Identity(name='amplitude')]


class ImageStimulus(Stimulus):

    def __init__(self, grid_coordinates, positive_only=True):
        self.grid_coordinates = pd.DataFrame(grid_coordinates, columns=['x', 'y'])

        if positive_only:
            self.bijectors = tfp.bijectors.Softplus(name='intensity')
        else:
            self.bijectors = tfp.bijectors.Identity(name='intensity')

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
