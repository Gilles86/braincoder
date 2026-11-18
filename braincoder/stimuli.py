import numpy as np
import pandas as pd
import logging
from keras import ops

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

                labels = self.dimension_labels
                if paradigm.shape[1] != len(labels):
                    labels = [f'dim_{ix}' for ix in range(paradigm.shape[1])]

                paradigm = pd.DataFrame(paradigm, columns=pd.Index(labels, name='stimulus dimensions'),
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
                stimulus = np.stack([ops.convert_to_numpy(bijector.forward(stimulus[:, ix])) for ix, bijector in enumerate(self.bijectors)], axis=1)
            else:
                stimulus = ops.convert_to_numpy(self.bijectors.forward(stimulus))

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


class OneDimensionalOrientationStimulus(Stimulus):
    """Stimulus for π-periodic orientation data (e.g. gabor gratings).

    Orientations in [0, π) — 0 rad and π rad represent the same grating,
    so the periodic wrap point is π, not 2π.
    """
    dimension_labels = ['x (radians, orientation)']

    def __init__(self):
        super().__init__()
        self.bijectors = [_Periodic(low=0.0, high=np.pi, name='x')]


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


GAIN_FIELD_COLUMN_LEVELS = ('kind', 'x | gain', 'y | location')


def gain_field_columns(grid_coordinates, n_locations, gain_labels):
    """Column index of a gain-field paradigm (see :func:`make_gain_field_paradigm`).

    ``('image', x, y)`` for every grid point, in grid order, followed by
    ``('field', gain_label, location_index)`` for every gain label (outer)
    and field location (inner).
    """
    grid_coordinates = np.asarray(grid_coordinates)
    image = [('image', float(x), float(y)) for x, y in grid_coordinates]
    field = [('field', str(label), int(ix)) for label in gain_labels for ix in range(n_locations)]
    return pd.MultiIndex.from_tuples(image + field, names=GAIN_FIELD_COLUMN_LEVELS)


class GainFieldImageStimulus(ImageStimulus):
    """An image on a pixel grid plus, per frame, the weight of every
    (gain label, field location) pair. Used by the gain-field PRF models."""

    def __init__(self, grid_coordinates, field_positions, gain_labels, positive_only=True):
        super().__init__(grid_coordinates, positive_only=positive_only)
        self.field_positions = np.asarray(field_positions, dtype=np.float32)
        self.gain_labels = list(gain_labels)
        self.n_grid = len(self.grid_coordinates)
        self.dimension_labels = gain_field_columns(
            self.grid_coordinates.values, len(self.field_positions), self.gain_labels)

    def clean_paradigm(self, paradigm):
        if paradigm is None:
            return None
        if not isinstance(paradigm, pd.DataFrame):
            paradigm = np.asarray(paradigm)
            if paradigm.ndim != 2:
                raise ValueError('A gain-field paradigm is 2D: (n_frames, n_grid + n_locations * n_gains). '
                                 'Build it with braincoder.stimuli.make_gain_field_paradigm.')
            self._check_n_columns(paradigm.shape[1])
            paradigm = pd.DataFrame(paradigm, columns=self.dimension_labels,
                                    index=pd.Index(np.arange(len(paradigm)), name='frame'))
        self._check_n_columns(paradigm.shape[1])
        return paradigm.astype(np.float32)

    def _check_n_columns(self, n_columns):
        if n_columns != len(self.dimension_labels):
            raise ValueError(
                f'The paradigm has {n_columns} columns, but this model needs '
                f'{self.n_grid} image columns + {len(self.field_positions)} locations x '
                f'{len(self.gain_labels)} gains = {len(self.dimension_labels)}. '
                'Build it with braincoder.stimuli.make_gain_field_paradigm.')


def make_gain_field_paradigm(images, assignments, field_positions, gain_labels, grid_coordinates=None,
                             index=None, atol=1e-4):
    """Build the paradigm of a gain-field PRF model.

    A gain-field model needs, for every frame (TR or trial), the stimulus
    image *and* which field locations carry which gain. This function puts
    both in one DataFrame.

    Parameters
    ----------
    images : array (n_frames, n_grid) or (n_frames, n_x, n_y), or DataFrame
        Stimulus intensity on the model's pixel grid, flattened in the same
        order as ``grid_coordinates``.
    assignments : array (n_frames, n_locations, n_gains), or list of dicts
        The weight W[t, l, k] of gain ``k`` at location ``l`` on frame ``t``
        (usually 0/1). As a list (one entry per frame), each entry maps a
        gain label to a location, a *list* of locations, or a ``{location:
        weight}`` dict. A location is an int index into ``field_positions``
        or an ``(x, y)`` *tuple* that must match one of them. Missing labels and
        ``None`` mean weight 0 (no modulation on that frame).
    field_positions : array (n_locations, 2)
        Every location a field can sit at, in grid units. If the attended or
        salient things move freely, pass all positions that occur.
    gain_labels : list of str
        Names of the gain types, e.g. ``['attended', 'ignored']``.
    grid_coordinates : array (n_grid, 2), optional
        The model's pixel coordinates, only used to label the image columns
        ``('image', x, y)``; without it they are ``('image', pixel_index, nan)``.
    index : pandas Index, optional
        Row index (defaults to the index of ``images`` if it is a DataFrame,
        else ``frame`` = 0..n_frames-1). Give it the index of your data.

    Returns
    -------
    DataFrame (n_frames, n_grid + n_gains * n_locations)
        Columns ``('image', x, y)`` for the pixels, then
        ``('field', gain_label, location_index)``.
    """
    field_positions = np.asarray(field_positions, dtype=np.float32)
    if field_positions.ndim != 2 or field_positions.shape[1] != 2:
        raise ValueError('field_positions must be (n_locations, 2).')
    gain_labels = [str(label) for label in gain_labels]
    if len(set(gain_labels)) != len(gain_labels):
        raise ValueError('gain_labels must be unique.')
    n_loc, n_gain = len(field_positions), len(gain_labels)

    if isinstance(images, pd.DataFrame):
        if index is None:
            index = images.index
        images = images.values
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images.reshape((images.shape[0], -1))
    if images.ndim != 2:
        raise ValueError('images must be (n_frames, n_grid) or (n_frames, n_x, n_y).')
    n_frames = len(images)

    def location_index(loc):
        if np.ndim(loc) == 0:
            ix = int(loc)
            if not 0 <= ix < n_loc:
                raise ValueError(f'Location index {ix} is outside 0..{n_loc - 1}.')
            return ix
        dist = np.abs(field_positions - np.asarray(loc, dtype=np.float32)[None, :]).max(1)
        ix = int(np.argmin(dist))
        if dist[ix] > atol:
            raise ValueError(f'Location {tuple(loc)} is not one of field_positions.')
        return ix

    if isinstance(assignments, np.ndarray) and assignments.ndim == 3:
        weights = assignments.astype(np.float32)
        if weights.shape != (n_frames, n_loc, n_gain):
            raise ValueError(f'assignments has shape {weights.shape}; expected '
                             f'(n_frames, n_locations, n_gains) = {(n_frames, n_loc, n_gain)}.')
    else:
        if len(assignments) != n_frames:
            raise ValueError(f'{len(assignments)} assignments for {n_frames} frames.')
        weights = np.zeros((n_frames, n_loc, n_gain), dtype=np.float32)
        for t, frame in enumerate(assignments):
            if frame is None:
                continue
            for label, locs in frame.items():
                if str(label) not in gain_labels:
                    raise ValueError(f'Unknown gain label {label!r}; gain_labels are {gain_labels}.')
                k = gain_labels.index(str(label))
                if locs is None:
                    continue
                if isinstance(locs, dict):
                    items = locs.items()
                elif np.ndim(locs) == 0 or isinstance(locs, tuple):
                    items = [(locs, 1.0)]
                else:
                    items = [(loc, 1.0) for loc in locs]
                for loc, w in items:
                    weights[t, location_index(loc), k] += w

    if index is None:
        index = pd.Index(np.arange(n_frames), name='frame')
    n_grid = images.shape[1]
    if grid_coordinates is not None:
        grid_coordinates = np.asarray(grid_coordinates)
        if len(grid_coordinates) != n_grid:
            raise ValueError(f'images have {n_grid} pixels but grid_coordinates has {len(grid_coordinates)}.')
        columns = gain_field_columns(grid_coordinates, n_loc, gain_labels)
    else:
        columns = pd.MultiIndex.from_tuples(
            [('image', float(i), np.nan) for i in range(n_grid)]
            + [('field', label, l) for label in gain_labels for l in range(n_loc)],
            names=GAIN_FIELD_COLUMN_LEVELS)
    # weights (T, L, K) -> columns ordered gain (outer), location (inner)
    field_block = np.transpose(weights, (0, 2, 1)).reshape(n_frames, n_gain * n_loc)
    return pd.DataFrame(np.concatenate([images, field_block], 1), index=index, columns=columns)
