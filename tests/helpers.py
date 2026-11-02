"""Shared test helpers."""
import numpy as np


class Bound:
    """A model plus the arguments a test keeps passing to it."""

    def __init__(self, model, paradigm=None, parameters=None, weights=None):
        self.model, self.paradigm, self.parameters, self.weights = model, paradigm, parameters, weights

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, paradigm=None, parameters=None, weights=None):
        return self.model.predict(self.paradigm if paradigm is None else paradigm,
                                  self.parameters if parameters is None else parameters,
                                  self.weights if weights is None else weights)

    def simulate(self, paradigm=None, parameters=None, weights=None, **kwargs):
        return self.model.simulate(self.paradigm if paradigm is None else paradigm,
                                   self.parameters if parameters is None else parameters,
                                   self.weights if weights is None else weights, **kwargs)


    def get_rf(self, **kwargs):
        return self.model.get_rf(self.parameters, **kwargs)


def grid_for(images):
    """Grid coordinates on [-1, 1]² for a (n_t, n_x, n_y) image paradigm."""
    grid = np.array(np.meshgrid(np.linspace(-1, 1, images.shape[1]),
                                np.linspace(-1, 1, images.shape[2])), dtype=np.float32)
    return np.reshape(np.swapaxes(grid, 2, 1), (2, -1)).T
