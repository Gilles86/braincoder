"""Gain-field PRF models: a spatial field that changes the stimulus drive
around chosen locations before the receptive field pools it.

The usual question: *does attending to (or ignoring, or expecting) something
at location l change how much the stimulus near l drives visual cortex?*
The model answers it on the stimulus grid. On frame t (a TR or a trial) the
image ``S_t(g)`` is multiplied by a modulation map ``M_t(g)``, and each
voxel's ordinary 2D PRF pools the result::

    F_t(g) = sum_fields sum_k sum_l  W[t, l, k] * gain_{field,k} * A_field(g; location l)
    M_t(g) = max(1 + F_t(g), 0)                   # 'multiplicative', clamp=True
    r_v(t) = baseline_v + amplitude_v * sum_g S_t(g) * M_t(g) * RF_v(g)

- ``W[t, l, k]`` (in the paradigm) says which locations carry gain type ``k``
  on frame ``t``, usually 0/1.
- ``A_field(g; l)`` is the field *shape* centred on location ``l``, peak 1
  (a :class:`GaussianField` by default; see :class:`FieldShape` for your own).
- ``gain_{field,k}`` and the shape parameters (e.g. the width) belong to the
  field, not to a voxel: fit them with ``shared_pars`` (see below).
- ``RF_v`` is the receptive field of the PRF model underneath, with its own
  amplitude convention unchanged (``GaussianPRF2D``,
  ``DifferenceOfGaussiansPRF2D``, or any image model via
  :class:`GainFieldMixin`).

With all gains at 0, the model is exactly the PRF model underneath.

Pieces
------
1. Field shape: :class:`FieldShape` subclasses map (offsets from the field
   location, shape parameters) to a field on the grid.
   :class:`GaussianField` and :class:`DoGField` ship.
2. Field spec: :class:`GainField` = name + shape + which gain labels it
   responds to + sign constraint of its gains. A model takes a list.
3. Combination rule: ``combination='multiplicative'`` (default, with
   ``clamp``) or any callable ``f(images, field_sums) -> images``; or override
   :meth:`GainFieldMixin._combine`.
4. PRF kernel: the class the mixin is combined with.

Parameter labels are the PRF labels, then for every field
``f'{field.name}_{shape_par}'`` and ``f'{field.name}_gain_{label}'``.
The default single field is called ``'field'``, so one Gaussian field with
gain labels ``['attended', 'ignored']`` gives ``field_sigma``,
``field_gain_attended``, ``field_gain_ignored``.

Field parameters are model-level
--------------------------------
The parameters DataFrame has one row per voxel, so the field columns are
repeated on every row and **must be identical across voxels** (the model
raises otherwise). Inside the computation the model uses their mean across
voxels, so it never builds a voxels x frames x pixels tensor. With
``ParameterFitter`` pass the field labels as ``shared_pars``
(:meth:`GainFieldMixin.field_parameter_labels` lists them), optionally with
the PRF labels as ``fixed_pars``.

Decoding
--------
``get_stimulus_pdf`` and friends are not meaningful for these models (the
paradigm is not a stimulus to decode). Decode with the PRF model underneath.
"""
import logging
import numpy as np
import pandas as pd
from keras import ops

from ..stimuli import GainFieldImageStimulus
from ..utils.backend import softplus_inverse
from .base import HRFEncodingModel
from .prf_2d import GaussianPRF2D, DifferenceOfGaussiansPRF2D


# ---------------------------------------------------------------------------
# Parameter transforms (unconstrained <-> native), column-wise
# ---------------------------------------------------------------------------

def _floor_softplus(floor):
    """``x = floor + softplus(raw)`` and its inverse."""
    return (lambda raw: floor + ops.softplus(raw),
            lambda x: softplus_inverse(x - floor))


_TRANSFORMS = {
    'identity': (lambda raw: raw, lambda x: x),
    'softplus': (ops.softplus, softplus_inverse),
    'negative_softplus': (lambda raw: -ops.softplus(raw), lambda x: softplus_inverse(-x)),
    'sigmoid': (ops.sigmoid, lambda x: ops.log(x / (1.0 - x))),
}


def _get_transform(t):
    if isinstance(t, tuple):
        return t
    if t not in _TRANSFORMS:
        raise ValueError(f'Unknown transform {t!r}; use one of {list(_TRANSFORMS)} '
                         'or a (forward, inverse) tuple.')
    return _TRANSFORMS[t]


# ---------------------------------------------------------------------------
# 1. Field shapes
# ---------------------------------------------------------------------------

class FieldShape:
    """Spatial profile of a field, centred on a field location.

    A new shape is a subclass with

    - ``parameter_labels``: names of its parameters (e.g. ``['sigma']``);
    - ``transformations``: one per parameter, ``'identity'``,
      ``'softplus'``, ``'sigmoid'`` or a ``(forward, inverse)`` tuple of
      ``keras.ops`` functions from unconstrained to native values;
    - :meth:`profile`.

    Example: an annulus (ring of radius ``radius`` and width ``width``)::

        class AnnulusField(FieldShape):
            parameter_labels = ['radius', 'width']
            transformations = ['softplus', 'softplus']

            def profile(self, dx, dy, radius, width):
                r = ops.sqrt(dx ** 2 + dy ** 2 + 1e-12)
                return ops.exp(-(r - radius) ** 2 / (2 * width ** 2))
    """

    parameter_labels = []
    transformations = []

    def profile(self, dx, dy, **parameters):
        """Field value at offsets ``(dx, dy)`` from the field location.

        ``dx, dy`` have shape ``(1, n_locations, n_grid)``; every parameter
        has shape ``(n_batches, 1, 1)``. Return ``(n_batches, n_locations,
        n_grid)``. Peak 1 keeps gains interpretable as the modulation at
        the field centre.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}()'


class GaussianField(FieldShape):
    """Isotropic Gaussian, peak 1: ``exp(-d^2 / (2 sigma^2))``.

    ``sigma = sigma_min + softplus(raw)``, in grid units.
    """

    parameter_labels = ['sigma']

    def __init__(self, sigma_min=1e-2):
        self.sigma_min = float(sigma_min)
        self.transformations = [_floor_softplus(self.sigma_min)]

    def profile(self, dx, dy, sigma):
        return ops.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))

    def __repr__(self):
        return f'GaussianField(sigma_min={self.sigma_min})'


class DoGField(FieldShape):
    """Difference of Gaussians ("Mexican hat"): a centre of width ``sigma``
    minus ``surround_weight`` times a surround of width
    ``surround_ratio * sigma``. Peak ``1 - surround_weight``.

    ``sigma = sigma_min + softplus``, ``surround_ratio = 1 + softplus``
    (surround wider than centre), ``surround_weight = sigmoid`` (0..1).
    """

    parameter_labels = ['sigma', 'surround_ratio', 'surround_weight']

    def __init__(self, sigma_min=1e-2):
        self.sigma_min = float(sigma_min)
        self.transformations = [_floor_softplus(self.sigma_min), _floor_softplus(1.0), 'sigmoid']

    def profile(self, dx, dy, sigma, surround_ratio, surround_weight):
        d2 = dx ** 2 + dy ** 2
        return (ops.exp(-d2 / (2.0 * sigma ** 2))
                - surround_weight * ops.exp(-d2 / (2.0 * (surround_ratio * sigma) ** 2)))

    def __repr__(self):
        return f'DoGField(sigma_min={self.sigma_min})'


# ---------------------------------------------------------------------------
# 2. Field spec
# ---------------------------------------------------------------------------

_GAIN_SIGNS = {'signed': 'identity', 'positive': 'softplus', 'negative': 'negative_softplus'}


class GainField:
    """One field: a shape, the gain labels it responds to, its gain sign.

    Parameters
    ----------
    name : str
        Prefix of its parameter labels (``f'{name}_sigma'``, ...).
    shape : FieldShape, default GaussianField()
    gain_labels : list of str or None
        Which of the model's gain labels this field responds to (``None`` =
        all). Each gives a parameter ``f'{name}_gain_{label}'``.
    gains : {'signed', 'positive', 'negative'}
        Constraint on the gains. ``'signed'`` (default) leaves them
        unbounded, so the data decide between enhancement (> 0) and
        suppression (< 0). With ``'positive'``/``'negative'`` do not
        initialise a gain at exactly 0.
    clamped : bool, default True
        With the default combination rule and ``clamp=True``, a clamped
        field sits inside the clip ``max(1 + F, 0)``; an unclamped one is
        added outside it (e.g. a brief transient on top of a sustained
        field: ``M = max(1 + F_sustained, 0) + F_transient``).
    """

    def __init__(self, name='field', shape=None, gain_labels=None, gains='signed', clamped=True):
        if gains not in _GAIN_SIGNS:
            raise ValueError(f'gains must be one of {list(_GAIN_SIGNS)}, got {gains!r}')
        self.name = str(name)
        self.shape = GaussianField() if shape is None else shape
        self.gain_labels = None if gain_labels is None else [str(l) for l in gain_labels]
        self.gains = gains
        self.clamped = bool(clamped)

    def __repr__(self):
        return (f'GainField(name={self.name!r}, shape={self.shape!r}, '
                f'gain_labels={self.gain_labels!r}, gains={self.gains!r}, clamped={self.clamped})')


# ---------------------------------------------------------------------------
# 3 + 4. The model
# ---------------------------------------------------------------------------

class GainFieldMixin:
    """Adds gain fields to any image PRF model (``GaussianPRF2D`` and
    subclasses). Put it first in the bases::

        class GainFieldDNPRF2D(GainFieldMixin, DivisiveNormalizationGaussianPRF2D):
            pass

    Parameters
    ----------
    grid_coordinates : array (n_grid, 2)
        Pixel coordinates, as for ``GaussianPRF2D``.
    field_positions : array (n_locations, 2)
        Every location a field can be centred on, in grid units. Any number,
        anywhere (they need not lie on the grid).
    gain_labels : list of str
        Names of the gain types, e.g. ``['attended', 'ignored']``. The
        paradigm says per frame which location carries which label, with
        which weight.
    fields : list of GainField, optional
        Default: one Gaussian field named ``'field'`` with signed gains for
        all labels.
    combination : 'multiplicative' or callable
        ``'multiplicative'`` (default)::

            M = max(1 + sum(clamped fields), 0) + sum(unclamped fields)
            S_eff = S * M

        (without ``clamp`` the ``max`` is dropped). A callable gets
        ``(images, field_sums)``, with ``field_sums`` one ``(n_batches,
        n_frames, n_grid)`` tensor per field in ``fields`` order, and
        returns the modulated images. Or override :meth:`_combine`.
    clamp : bool, default True
        Clip the clamped part of the modulation at 0, so that strong
        suppression cannot invert the stimulus.
    image_scales : dict {name: mask}, optional
        Extra model-level parameters that scale part of the image before
        the fields act: ``S <- S * (1 + sum_s (scale_s - 1) * mask_s)``.
        ``mask`` is ``(n_grid,)`` (bool or weights). Each adds a positive
        parameter ``name`` (softplus); its null value is 1. Use it when
        two kinds of stimulus share one image but their relative drive is
        unknown (e.g. a mapping bar and task items).
    ties : dict {label: tie}, optional
        Parameters computed from others instead of being free. ``tie`` is a
        number (constant), another label (equal to it), or ``{label: coef}``
        (linear combination of native values). Examples: a zero-sum pair
        ``{'field_gain_b': {'field_gain_a': -1/3}}``, two fields with one
        width ``{'b_sigma': 'a_sigma'}``, a switched-off gain
        ``{'field_gain_c': 0.0}``. Tied columns ignore their raw value (no
        gradient); their native value is filled in by the forward
        transform, so predictions and fitted parameters always satisfy the
        tie.
    **kwargs
        Passed to the PRF model (e.g. ``rf_normalization``).
    """

    def __init__(self, grid_coordinates=None, field_positions=None, gain_labels=None, fields=None,
                 combination='multiplicative', clamp=True, image_scales=None, ties=None,
                 positive_image_values_only=True, verbosity=logging.INFO, **kwargs):

        if field_positions is None:
            raise ValueError(f'{type(self).__name__} needs field_positions: (n_locations, 2).')
        if gain_labels is None or len(gain_labels) == 0:
            raise ValueError(f'{type(self).__name__} needs gain_labels, e.g. ["attended"].')

        field_positions = np.asarray(field_positions, dtype=np.float32)
        if field_positions.ndim != 2 or field_positions.shape[1] != 2:
            raise ValueError('field_positions must be (n_locations, 2).')
        self.field_positions = field_positions
        self.gain_labels = [str(l) for l in gain_labels]
        if len(set(self.gain_labels)) != len(self.gain_labels):
            raise ValueError('gain_labels must be unique.')

        self.fields = [GainField()] if fields is None else list(fields)
        if not callable(combination) and combination != 'multiplicative':
            raise ValueError("combination must be 'multiplicative' or a callable.")
        self.combination = combination
        self.clamp = clamp

        super().__init__(grid_coordinates=grid_coordinates,
                         positive_image_values_only=positive_image_values_only,
                         verbosity=verbosity, **kwargs)

        self.stimulus = GainFieldImageStimulus(self.grid_coordinates, self.field_positions,
                                               self.gain_labels, positive_only=positive_image_values_only)
        n_grid = len(self.grid_coordinates)

        # The PRF labels come from the kernel class; append the field labels.
        self._prf_labels = list(type(self).parameter_labels)
        self._field_specs = []         # per field: (shape_labels, gain_par_labels, gain_label_ixs)
        field_labels, field_transforms = [], []
        for field in self.fields:
            labels_f = field.gain_labels if field.gain_labels is not None else self.gain_labels
            unknown = [l for l in labels_f if l not in self.gain_labels]
            if unknown:
                raise ValueError(f'Field {field.name!r} uses unknown gain labels {unknown}.')
            if len(field.shape.parameter_labels) != len(field.shape.transformations):
                raise ValueError(f'{field.shape!r}: one transformation per parameter label.')
            shape_labels = [f'{field.name}_{p}' for p in field.shape.parameter_labels]
            gain_par_labels = [f'{field.name}_gain_{l}' for l in labels_f]
            field_labels += shape_labels + gain_par_labels
            field_transforms += ([_get_transform(t) for t in field.shape.transformations]
                                 + [_get_transform(_GAIN_SIGNS[field.gains])] * len(labels_f))
            self._field_specs.append((shape_labels, gain_par_labels,
                                      [self.gain_labels.index(l) for l in labels_f]))

        self.image_scales = {}
        for name, mask in (image_scales or {}).items():
            mask = np.asarray(mask, dtype=np.float32).ravel()
            if mask.shape != (n_grid,):
                raise ValueError(f'image_scales[{name!r}] must have one value per grid point ({n_grid}).')
            self.image_scales[str(name)] = mask
            field_labels.append(str(name))
            field_transforms.append(_get_transform('softplus'))

        labels = self._prf_labels + field_labels
        if len(set(labels)) != len(labels):
            raise ValueError(f'Parameter labels are not unique: {labels}. Give the fields different names.')
        self._field_labels = field_labels
        self._field_transforms = field_transforms
        self.parameter_labels = labels

        self.ties = {}
        for label, tie in (ties or {}).items():
            if label not in labels:
                raise ValueError(f'Cannot tie unknown parameter {label!r}.')
            if isinstance(tie, str):
                tie = {tie: 1.0}
            if isinstance(tie, dict):
                for src in tie:
                    if src not in labels:
                        raise ValueError(f'Tie of {label!r} refers to unknown parameter {src!r}.')
                tie = {str(k): float(v) for k, v in tie.items()}
            else:
                tie = float(tie)
            self.ties[label] = tie
        for label, tie in self.ties.items():
            if isinstance(tie, dict) and any(src in self.ties for src in tie):
                raise ValueError(f'Tie of {label!r} refers to another tied parameter; tie to free ones.')

    # -------------------------------------------------------------- helpers
    def field_parameter_labels(self, include_tied=True):
        """Labels of the model-level parameters (fields, image scales):
        pass them as ``shared_pars``."""
        return [l for l in self._field_labels if include_tied or l not in self.ties]

    def prf_parameter_labels(self):
        """Labels of the per-voxel PRF parameters."""
        return list(self._prf_labels)

    def null_field_parameters(self):
        """Null value of every field parameter that has one: gains 0, image scales 1."""
        out = {}
        for label in self._field_labels:
            if label in self.image_scales:
                out[label] = 1.0
            elif any(label in g for _, g, _ in self._field_specs):
                out[label] = 0.0
        return out

    def make_paradigm(self, images, assignments, index=None):
        """:func:`braincoder.stimuli.make_gain_field_paradigm` for this model."""
        from ..stimuli import make_gain_field_paradigm
        return make_gain_field_paradigm(images, assignments, self.field_positions, self.gain_labels,
                                        grid_coordinates=self.grid_coordinates.values, index=index)

    def add_field_parameters(self, prf_parameters, **values):
        """Append the field columns to per-voxel PRF parameters.

        Gains default to 0 and image scales to 1 (the null model, which
        predicts exactly what the PRF model underneath predicts); shape
        parameters such as widths need a value::

            pars = model.add_field_parameters(prf_pars, field_sigma=1.0)
        """
        pars = prf_parameters[self._prf_labels].copy()
        defaults = self.null_field_parameters()
        for label in self._field_labels:
            if label in values:
                pars[label] = np.float32(values.pop(label))
            elif label in defaults:
                pars[label] = np.float32(defaults[label])
            elif label in self.ties:
                pars[label] = np.float32(np.nan)       # filled in below
            else:
                raise ValueError(f'add_field_parameters needs a value for {label!r}.')
        if values:
            raise ValueError(f'Unknown field parameters: {list(values)}; known: {self._field_labels}.')
        return self.apply_ties(pars)

    def apply_ties(self, parameters):
        """Return a copy of a parameter DataFrame with the tied columns filled in."""
        parameters = parameters.copy()
        for label, tie in self.ties.items():
            if isinstance(tie, dict):
                parameters[label] = sum(c * parameters[src] for src, c in tie.items()).astype(np.float32)
            else:
                parameters[label] = np.float32(tie)
        return parameters

    def _get_parameters(self, parameters=None):
        parameters = super()._get_parameters(parameters)
        if parameters is not None and self.ties:
            parameters = self.apply_ties(parameters)
        if parameters is not None and len(parameters) > 1:
            free = self.field_parameter_labels(include_tied=False)
            vals = parameters[free].values.astype(np.float64)
            vals = vals[np.isfinite(vals).all(1)]
            if len(vals):
                spread = np.ptp(vals, axis=0) > 1e-5 * (1 + np.abs(vals).max(0))
                if spread.any():
                    bad = [l for l, b in zip(free, spread) if b]
                    raise ValueError(
                        f'The field parameters {bad} differ between voxels. They are model-level: '
                        'use the same value on every row (fit them with ParameterFitter(..., '
                        'shared_pars=model.field_parameter_labels())).')
        return parameters

    # ------------------------------------------------------- computation
    def _shared_values(self, parameters):
        """Model-level parameters (B, P): mean over the finite voxel rows
        (which are identical; see the module docstring)."""
        finite = ops.cast(ops.all(ops.isfinite(parameters), axis=2, keepdims=True), 'float32')
        safe = ops.where(finite > 0, parameters, ops.zeros_like(parameters))
        return ops.sum(safe * finite, axis=1) / ops.maximum(ops.sum(finite, axis=1), 1.0)

    def _field_sums(self, field_weights, shared):
        """One F_t(g) (n_batches, n_frames, n_grid) per field.

        ``field_weights``: (B, T, n_gains * n_locations), gain-major.
        ``shared``: (B, P) native model-level values.
        """
        n_loc, n_gain = len(self.field_positions), len(self.gain_labels)
        W = ops.reshape(field_weights,
                        (ops.shape(field_weights)[0], ops.shape(field_weights)[1], n_gain, n_loc))

        grid = ops.convert_to_tensor(self._grid_coordinates, dtype='float32')
        pos = ops.convert_to_tensor(self.field_positions, dtype='float32')
        dx = (grid[None, :, 0] - pos[:, 0, None])[None]     # (1, L, G)
        dy = (grid[None, :, 1] - pos[:, 1, None])[None]

        sums = []
        for field, (shape_labels, gain_par_labels, gain_ixs) in zip(self.fields, self._field_specs):
            shape_pars = {p: shared[:, self.parameter_labels.index(l), None, None]
                          for p, l in zip(field.shape.parameter_labels, shape_labels)}
            A = field.shape.profile(dx, dy, **shape_pars)                         # (B, L, G)
            w = 0.0
            for k, label in zip(gain_ixs, gain_par_labels):
                gain = shared[:, self.parameter_labels.index(label), None, None]  # (B, 1, 1)
                w = w + W[:, :, k, :] * gain                                      # (B, T, L)
            sums.append(ops.einsum('btl,blg->btg', w, A))
        return sums

    def _scale_images(self, images, shared):
        if not self.image_scales:
            return images
        factor = 1.0
        for name, mask in self.image_scales.items():
            scale = shared[:, self.parameter_labels.index(name), None, None]       # (B, 1, 1)
            factor = factor + (scale - 1.0) * ops.convert_to_tensor(mask)[None, None, :]
        return images * factor

    def _combine(self, images, field_sums):
        """Modulated images from the images and one field sum per field (all (B, T, G))."""
        if callable(self.combination):
            return self.combination(images, field_sums)
        inside, outside = 1.0, 0.0
        for field, F in zip(self.fields, field_sums):
            if field.clamped:
                inside = inside + F
            else:
                outside = outside + F
        if self.clamp:
            inside = ops.maximum(inside, 0.0)
        return images * (inside + outside)

    def _modulated_images(self, paradigm, parameters):
        G = self._grid_coordinates.shape[0]
        shared = self._shared_values(parameters)
        images = self._scale_images(paradigm[..., :G], shared)
        return self._combine(images, self._field_sums(paradigm[..., G:], shared))

    def get_modulated_images(self, paradigm, parameters):
        """The images as the PRFs see them, ``S_t(g) * M_t(g)``: DataFrame frames x pixels."""
        paradigm = self.get_paradigm(paradigm)
        parameters = self._get_parameters(parameters)
        p = ops.convert_to_tensor(paradigm.values[None], dtype='float32')
        pars = ops.convert_to_tensor(parameters.values[None], dtype='float32')
        out = self._modulated_images(p, pars)
        return pd.DataFrame(ops.convert_to_numpy(out)[0], index=paradigm.index,
                            columns=pd.MultiIndex.from_frame(self.grid_coordinates))

    def _basis_predictions(self, paradigm, parameters):
        return super()._basis_predictions(self._modulated_images(paradigm, parameters), parameters)

    # ------------------------------------------------------- transforms
    def _n_hrf(self):
        return len(self.hrf_model.parameter_labels) if getattr(self, 'flexible_hrf_parameters', False) else 0

    def _transform_parameters_forward(self, parameters):
        n_prf, n_field, n_hrf = len(self._prf_labels), len(self._field_labels), self._n_hrf()
        cols = [super()._transform_parameters_forward(parameters[:, :n_prf])]
        cols += [t[0](parameters[:, n_prf + i][:, None]) for i, t in enumerate(self._field_transforms)]
        if n_hrf:
            cols.append(self.hrf_model._transform_parameters_forward(parameters[:, n_prf + n_field:]))
        native = ops.concatenate(cols, axis=1)
        if not self.ties:
            return native
        labels = self.parameter_labels
        out = [native[:, i][:, None] for i in range(len(labels))]
        for label, tie in self.ties.items():
            if isinstance(tie, dict):
                value = sum(c * native[:, labels.index(src)][:, None] for src, c in tie.items())
            else:
                value = ops.full_like(native[:, :1], tie)
            out[labels.index(label)] = value
        return ops.concatenate(out, axis=1)

    def _transform_parameters_backward(self, parameters):
        n_prf, n_field, n_hrf = len(self._prf_labels), len(self._field_labels), self._n_hrf()
        labels = self.parameter_labels
        cols = [super()._transform_parameters_backward(parameters[:, :n_prf])]
        for i, t in enumerate(self._field_transforms):
            col = parameters[:, n_prf + i][:, None]
            # A tied column's raw value is unused; 0 keeps it finite.
            cols.append(ops.zeros_like(col) if labels[n_prf + i] in self.ties else t[1](col))
        if n_hrf:
            cols.append(self.hrf_model._transform_parameters_backward(parameters[:, n_prf + n_field:]))
        raw = ops.concatenate(cols, axis=1)
        tied_prf = [labels.index(l) for l in self.ties if labels.index(l) < n_prf]
        if tied_prf:
            raw = ops.concatenate([ops.zeros_like(raw[:, i][:, None]) if i in tied_prf else raw[:, i][:, None]
                                   for i in range(len(labels))], axis=1)
        return raw

    # ------------------------------------------------- not meaningful here
    def to_linear_model(self, parameters):
        raise NotImplementedError('A gain-field model is not linear in a fixed image basis; '
                                  'use the PRF model underneath.')

    def __repr__(self):
        return (f'{type(self).__name__}(n_grid={len(self.grid_coordinates)}, '
                f'n_locations={len(self.field_positions)}, gain_labels={self.gain_labels}, '
                f'fields={self.fields})')


class GainFieldPRF2D(GainFieldMixin, GaussianPRF2D):
    """Gaussian 2D PRF with gain fields. See :mod:`braincoder.models.gain_field`.

    One line for the common case (one Gaussian field, signed gains)::

        model = GainFieldPRF2D(grid_coordinates, field_positions, ['attended', 'ignored'])
        model.parameter_labels
        # ['x', 'y', 'sd', 'baseline', 'amplitude',
        #  'field_sigma', 'field_gain_attended', 'field_gain_ignored']
    """


class GainFieldDoGPRF2D(GainFieldMixin, DifferenceOfGaussiansPRF2D):
    """Difference-of-Gaussians 2D PRF with gain fields."""


def _with_hrf_init(base):
    def __init__(self, grid_coordinates=None, field_positions=None, gain_labels=None, fields=None,
                 combination='multiplicative', clamp=True, image_scales=None, ties=None,
                 hrf_model=None, flexible_hrf_parameters=False,
                 positive_image_values_only=True, verbosity=logging.INFO, **kwargs):
        base.__init__(self, grid_coordinates=grid_coordinates, field_positions=field_positions,
                      gain_labels=gain_labels, fields=fields, combination=combination, clamp=clamp,
                      image_scales=image_scales, ties=ties,
                      positive_image_values_only=positive_image_values_only, verbosity=verbosity, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters)
    return __init__


class GainFieldPRF2DWithHRF(HRFEncodingModel, GainFieldPRF2D):
    """:class:`GainFieldPRF2D` for BOLD time courses (modulation before the HRF)."""
    __init__ = _with_hrf_init(GainFieldPRF2D)


class GainFieldDoGPRF2DWithHRF(HRFEncodingModel, GainFieldDoGPRF2D):
    """:class:`GainFieldDoGPRF2D` for BOLD time courses (modulation before the HRF)."""
    __init__ = _with_hrf_init(GainFieldDoGPRF2D)
