"""Stimulus parameterisations: decode a few stimulus parameters instead of an image.

An encoding model ``f(s; θ)`` is defined on a stimulus space S, for a pRF model the
pixels of an image. Many experiments only show a small family of images, which a
few numbers describe: a moving bar is fixed by the position of its centre line
along the normal, the orientation of that normal and its width. A
*stimulus parameterisation* is the map

    g: Φ → S,   φ ↦ image,

and decoding then targets the posterior over φ instead of over images:

    p(φ | y) ∝ p(y | f(g(φ); θ)) p(φ).

The encoding model is unchanged; only the candidate set (a grid over φ, or a
continuous φ for gradient methods) is generated through ``g``. Over a few
dimensions the posterior is a low-dimensional integral, so it can be computed
exactly on a grid, which is what :meth:`EncodingModel.get_stimulus_pdf` already
does for a list of candidate images.

Contents
--------
ParametricStimulus
    Base class: named parameters, bounds, a differentiable ``to_image`` in
    ``keras.ops`` (TensorFlow, JAX and PyTorch), grids and candidate sets.
BarStimulus
    A bar: hard-edged or soft-edged (differentiable), optional contrast
    (presence), optional aperture mask.
get_parametric_stimulus_pdf
    Grid posterior over φ (plus an optional blank state) through
    ``get_stimulus_pdf``, with a prior.
marginalize
    Marginal posterior over a subset of parameters.
forward_backward
    Exact posterior marginals of a hidden Markov model over a candidate set
    (e.g. φ grid over time, with a Markov prior on the bar's motion).
ParametricStimulusFitter (in ``braincoder.optimize``)
    Gradient-based MAP over φ, per volume or, with a ``*WithHRF`` model, for a
    whole time series.

Orientation convention: ``orientation`` is the direction of the bar's *normal*
in degrees (0 = vertical bar, moving horizontally), ``position`` the signed
distance of the bar's centre line from the origin along that normal. The bar
with (orientation + 180, -position) is the same image, so g is two-to-one on
the full circle. Grids use orientation in [0, 180) with a signed position,
which has no duplicates; continuous optimisation uses the full circle (no wrap
boundary) and :meth:`BarStimulus.canonicalize` maps results back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from keras import ops

__all__ = ['ParametricStimulus', 'BarStimulus', 'get_parametric_stimulus_pdf', 'marginalize',
           'forward_backward']


def _as_grid(grid_coordinates):
    g = np.asarray(grid_coordinates.values if hasattr(grid_coordinates, 'values') else grid_coordinates, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError('grid_coordinates must be (n_pixels, 2): x and y of every pixel')
    return g


class ParametricStimulus(object):
    """A map ``g`` from a few named parameters to an image on a pixel grid.

    Subclasses set ``parameter_labels`` and implement :meth:`_render` (keras ops,
    differentiable where the subclass says so) and, optionally,
    :meth:`_render_numpy` (exact numpy rendering for candidate sets).

    Parameters
    ----------
    grid_coordinates : (n_pixels, 2) array or DataFrame
        x and y of every pixel, in the same order as the encoding model's
        ``grid_coordinates``.
    aperture : (n_pixels,) array, optional
        Multiplies every image (e.g. a circular field, or a 4° aperture).
    """

    parameter_labels = []

    def __init__(self, grid_coordinates, aperture=None):
        self.grid_coordinates = _as_grid(grid_coordinates)
        self.n_pixels = len(self.grid_coordinates)
        if aperture is not None:
            aperture = np.asarray(aperture, dtype=np.float32).ravel()
            if len(aperture) != self.n_pixels:
                raise ValueError(f'aperture has {len(aperture)} pixels, grid {self.n_pixels}')
        self.aperture = aperture
        self._x = self.grid_coordinates[:, 0].astype(np.float32)
        self._y = self.grid_coordinates[:, 1].astype(np.float32)

    # -- parameters ------------------------------------------------------------

    @property
    def n_parameters(self):
        return len(self.parameter_labels)

    def _as_array(self, phi):
        """(n, n_parameters) numpy array from a DataFrame (by label) or an array."""
        if isinstance(phi, pd.DataFrame):
            missing = [l for l in self.parameter_labels if l not in phi.columns]
            if missing:
                raise ValueError(f'parameters {missing} missing; need {self.parameter_labels}')
            phi = phi[self.parameter_labels].values
        phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
        if phi.shape[1] != self.n_parameters:
            raise ValueError(f'phi has {phi.shape[1]} columns, need {self.n_parameters} ({self.parameter_labels})')
        return phi

    def to_unconstrained(self, phi):
        """Parameters → unconstrained space (numpy). Default: identity."""
        return self._as_array(phi).astype(np.float32)

    def from_unconstrained(self, u):
        """Unconstrained (tensor, n x n_parameters) → parameters (tensor). Default: identity."""
        return u

    def canonicalize(self, phi):
        """Map equivalent parameter values to one representative. Default: unchanged."""
        return phi

    # -- rendering -------------------------------------------------------------

    def _render(self, phi):
        raise NotImplementedError

    def _render_numpy(self, phi):
        return np.asarray(ops.convert_to_numpy(self._render(ops.convert_to_tensor(phi.astype(np.float32)))))

    def to_image(self, phi):
        """Images of ``phi`` (tensor or array, n x n_parameters) as a tensor (n, n_pixels), in keras ops."""
        phi = ops.convert_to_tensor(phi, dtype='float32')
        im = self._render(phi)
        if self.aperture is not None:
            im = im * ops.convert_to_tensor(self.aperture)[None, :]
        return im

    def images(self, phi):
        """Images of ``phi`` (DataFrame or array) as a float32 numpy array (n, n_pixels).

        NaN rows (a blank, see :meth:`candidates`) give an empty image.
        """
        phi = self._as_array(phi)
        blank = np.isnan(phi).any(1)
        out = np.zeros((len(phi), self.n_pixels), dtype=np.float32)
        if (~blank).any():
            im = self._render_numpy(phi[~blank])
            if self.aperture is not None:
                im = im * self.aperture[None, :]
            out[~blank] = im
        return out

    def __call__(self, phi):
        """Same as :meth:`images`; lets the object serve as a ``render`` function."""
        return self.images(phi)

    # -- grids -----------------------------------------------------------------

    def grid(self, **values):
        """Full factorial grid over φ as a DataFrame with columns ``parameter_labels``.

        Every parameter needs a list of values, e.g. ``grid(orientation=np.arange(0, 180, 15),
        position=np.arange(-8, 8.01, .5), width=[1., 2.])``.
        """
        missing = [l for l in self.parameter_labels if l not in values]
        if missing:
            raise ValueError(f'grid needs values for {missing}')
        idx = pd.MultiIndex.from_product([np.asarray(values[l], dtype=np.float64) for l in self.parameter_labels],
                                         names=self.parameter_labels)
        return idx.to_frame(index=False)

    def candidates(self, grid, include_blank=False, min_area=None):
        """Candidate set (table, images) for decoding from a grid over φ.

        Parameters
        ----------
        grid : DataFrame
            Rows of φ (e.g. from :meth:`grid`).
        include_blank : bool
            Prepend an empty image as row 0, with NaN parameters: the
            presence state "no stimulus", where φ is undefined.
        min_area : float, optional
            Drop candidates whose image sum is below ``min_area`` times the
            median image sum (stimuli almost entirely outside the field).

        Returns
        -------
        table : DataFrame, one row per candidate, columns ``parameter_labels``
        images : float32 array (n_candidates, n_pixels)
        """
        grid = pd.DataFrame(grid)[self.parameter_labels].astype(np.float64).reset_index(drop=True)
        ims = self.images(grid)
        if min_area is not None:
            area = ims.sum(1)
            keep = area > min_area * np.median(area)
            grid, ims = grid[keep].reset_index(drop=True), ims[keep]
        if include_blank:
            blank = pd.DataFrame(np.full((1, self.n_parameters), np.nan), columns=self.parameter_labels)
            grid = pd.concat([blank, grid], ignore_index=True)
            ims = np.concatenate([np.zeros((1, self.n_pixels), np.float32), ims])
        return grid, ims


class BarStimulus(ParametricStimulus):
    """A bar of given orientation, position and width, optionally with a contrast.

    ``image(x, y) = contrast · edge(width / 2 − |x cos θ + y sin θ − position|) · aperture``

    with θ = ``orientation`` (degrees, direction of the bar's normal).

    Parameters
    ----------
    grid_coordinates : (n_pixels, 2)
    edge : 'hard' or 'soft'
        'hard': 1 inside the bar, 0 outside (``|d| <= width / 2``, as a binary
        stimulus aperture); piecewise constant, so no useful gradient.
        'soft': a logistic edge, ``sigmoid((width / 2 − |d|) / edge_width)``,
        differentiable in every parameter (``|d|`` is smoothed at 0).
    edge_width : float, optional
        Logistic scale of the soft edge in the units of the grid. Default: a
        quarter of the pixel spacing, so the edge is within about one pixel.
    contrast : bool
        Add a ``contrast`` parameter in [0, 1] that multiplies the image. With
        a contrast, "no bar" is the point contrast = 0, where orientation,
        position and width are not identified; a discrete blank state
        (``include_blank`` in :meth:`candidates`) is usually the better model
        of blank frames.
    aperture : (n_pixels,) array, optional
        Mask that multiplies the bar (e.g. a bar seen only inside a 4° aperture).
    max_width : float, optional
        Upper bound on width for continuous optimisation (default: the field radius).
    min_width : float, optional
        Lower bound on width for continuous optimisation (default: half a pixel).
    """

    def __init__(self, grid_coordinates, edge='hard', edge_width=None, contrast=False, aperture=None,
                 max_width=None, min_width=None):
        super().__init__(grid_coordinates, aperture=aperture)
        if edge not in ('hard', 'soft'):
            raise ValueError("edge must be 'hard' or 'soft'")
        self.edge = edge
        xs = np.unique(np.round(self.grid_coordinates[:, 0], 6))
        self.pixel_spacing = float(np.diff(xs).min()) if len(xs) > 1 else 1.
        self.edge_width = float(self.pixel_spacing / 4. if edge_width is None else edge_width)
        self.contrast = contrast
        self.parameter_labels = ['orientation', 'position', 'width'] + (['contrast'] if contrast else [])
        self.radius = float(np.sqrt((self.grid_coordinates ** 2).sum(1)).max())
        self.max_width = float(self.radius if max_width is None else max_width)
        self.min_width = float(self.pixel_spacing / 2. if min_width is None else min_width)

    # distance of every pixel to the bar's centre line, keras ops
    def _distance(self, phi):
        th = phi[:, 0:1] * (np.pi / 180.)
        x = ops.convert_to_tensor(self._x)[None, :]
        y = ops.convert_to_tensor(self._y)[None, :]
        return x * ops.cos(th) + y * ops.sin(th) - phi[:, 1:2]

    def _render(self, phi):
        d = self._distance(phi)
        half = phi[:, 2:3] / 2.
        if self.edge == 'hard':
            im = ops.cast(ops.less_equal(ops.abs(d), half), 'float32')
        else:
            ad = ops.sqrt(d ** 2 + (.1 * self.edge_width) ** 2)
            im = ops.sigmoid((half - ad) / self.edge_width)
        if self.contrast:
            im = im * phi[:, 3:4]
        return im

    def _render_numpy(self, phi):
        if self.edge == 'soft':
            return super()._render_numpy(phi)
        # Hard edge in float64, so that candidate images are exactly those of a binary aperture
        # rendered with the same formula (no float32 ties at the edge).
        t = np.deg2rad(phi[:, 0])[:, None]
        x = self.grid_coordinates[:, 0][None, :]
        y = self.grid_coordinates[:, 1][None, :]
        im = (np.abs(x * np.cos(t) + y * np.sin(t) - phi[:, 1:2]) <= phi[:, 2:3] / 2.).astype(np.float32)
        if self.contrast:
            im = im * phi[:, 3:4].astype(np.float32)
        return im

    # -- continuous parameters -------------------------------------------------

    def to_unconstrained(self, phi):
        """orientation, position: identity (orientation is periodic in g itself); width: logit between
        min_width and max_width; contrast: logit."""
        phi = self._as_array(phi).copy()
        lo, hi = self.min_width, self.max_width
        w = np.clip((phi[:, 2] - lo) / (hi - lo), 1e-4, 1 - 1e-4)
        phi[:, 2] = np.log(w / (1 - w))
        if self.contrast:
            c = np.clip(phi[:, 3], 1e-4, 1 - 1e-4)
            phi[:, 3] = np.log(c / (1 - c))
        return phi.astype(np.float32)

    def from_unconstrained(self, u):
        parts = [u[:, 0:1], u[:, 1:2], self.min_width + (self.max_width - self.min_width) * ops.sigmoid(u[:, 2:3])]
        if self.contrast:
            parts.append(ops.sigmoid(u[:, 3:4]))
        return ops.concatenate(parts, axis=1)

    def canonicalize(self, phi):
        """Orientation into [0, 180), flipping the sign of position where it moved by 180°."""
        df = phi.copy() if isinstance(phi, pd.DataFrame) else pd.DataFrame(self._as_array(phi), columns=self.parameter_labels)
        th = np.mod(df['orientation'].values, 360.)
        flip = th >= 180.
        df['orientation'] = np.where(flip, th - 180., th)
        df['position'] = np.where(flip, -df['position'].values, df['position'].values)
        return df

    @staticmethod
    def centre(phi):
        """Foot point of the bar's centre line, (position cos θ, position sin θ): invariant to the 180° symmetry."""
        t = np.deg2rad(np.asarray(phi['orientation'], dtype=np.float64))
        p = np.asarray(phi['position'], dtype=np.float64)
        return np.stack([p * np.cos(t), p * np.sin(t)], 1)


# ---------------------------------------------------------------------------
# decoding on a grid over φ
# ---------------------------------------------------------------------------

def get_parametric_stimulus_pdf(model, data, stimulus, grid, parameters=None, weights=None, noise_model=None,
                                omega=None, dof=None, temperature=None, log_prior=None, include_blank=False,
                                blank_prior=None, min_area=None):
    """Posterior over a grid of stimulus parameters, per row of ``data``.

    The candidate images are ``stimulus.candidates(grid)``; the likelihood is
    the one of :meth:`EncodingModel.get_stimulus_pdf` (same noise models,
    temperature); the prior is ``log_prior`` over the grid rows (default flat)
    and, with ``include_blank``, a discrete blank state with prior
    probability ``blank_prior`` (the rest spread over the grid by
    ``log_prior``). Because the grid enumerates φ, the result is the exact
    posterior on that grid; marginals over parameters are sums
    (:func:`marginalize`).

    Note: a grid is a quadrature of the posterior over continuous φ. Its
    spacing should be small against the posterior width, and a flat prior
    over φ is not flat over images (it is the prior of this parameterisation).

    Parameters
    ----------
    model : EncodingModel (not ``*WithHRF``: decode single responses or betas)
    data : (n, n_voxels) DataFrame or array
    stimulus : ParametricStimulus
    grid : DataFrame of φ rows
    log_prior : (n_grid,) array, optional
        Log prior over the grid rows (need not be normalised).
    include_blank : bool
    blank_prior : float in (0, 1), required with ``include_blank``

    Returns
    -------
    DataFrame (n, n_candidates), rows sum to 1; columns are a MultiIndex of
    the parameter values (the blank, if any, first, with NaN values).
    """
    if include_blank and (blank_prior is None or not 0 < blank_prior < 1):
        raise ValueError('include_blank needs blank_prior in (0, 1)')
    table, images = stimulus.candidates(grid, include_blank=include_blank, min_area=min_area)
    n_bar = len(table) - int(include_blank)
    if log_prior is None:
        lp_bar = np.zeros(n_bar)
    else:
        lp_bar = np.asarray(log_prior, dtype=np.float64)
        if min_area is not None:
            raise ValueError('pass log_prior only without min_area (rows would no longer align)')
        if len(lp_bar) != n_bar:
            raise ValueError(f'log_prior has {len(lp_bar)} entries, grid {n_bar}')
    lp_bar = lp_bar - np.logaddexp.reduce(lp_bar)
    if include_blank:
        lp = np.r_[np.log(blank_prior), np.log1p(-blank_prior) + lp_bar]
    else:
        lp = lp_bar
    lik = model.get_stimulus_pdf(data, images, parameters=parameters, weights=weights, omega=omega, dof=dof,
                                 noise_model=noise_model, temperature=temperature, normalize=False)
    with np.errstate(divide='ignore'):
        a = np.log(np.asarray(lik.values, dtype=np.float64)) + lp[None, :]
    a = a - a.max(1, keepdims=True)
    post = np.exp(a)
    post /= post.sum(1, keepdims=True)
    return pd.DataFrame(post, index=lik.index, columns=pd.MultiIndex.from_frame(table))


def marginalize(pdf, labels):
    """Marginal posterior over ``labels`` (a parameter name or a list) from a grid posterior.

    The blank state (NaN parameters) is kept as its own column (NaN).
    """
    labels = [labels] if isinstance(labels, str) else list(labels)
    cols = pdf.columns.to_frame(index=False)[labels]
    key = pd.MultiIndex.from_frame(cols) if len(labels) > 1 else pd.Index(cols[labels[0]], name=labels[0])
    out = pdf.T.groupby(key, dropna=False, sort=True).sum().T
    return out


def forward_backward(log_emission, log_transition, log_initial=None):
    """Exact posterior marginals of a hidden Markov model (log-space forward–backward).

    For a stimulus that changes slowly over time (a sweeping bar), a Markov
    prior over a candidate set turns per-frame evidence into a posterior over
    the whole sequence whose frame marginals this computes exactly, at a cost
    of O(n K²). Use it on emissions that are conditionally independent over
    frames: e.g. per-frame scores of a deconvolved (HRF-removed) image
    sequence, not raw BOLD volumes, whose overlapping HRFs couple frames.

    Parameters
    ----------
    log_emission : (n, K) array
        Log p(evidence_t | state k), up to a constant per frame.
    log_transition : (K, K) array
        Log p(state_{t+1} = j | state_t = i), rows i.
    log_initial : (K,) array, optional
        Log prior of the first state (default: stationary-free uniform).

    Returns
    -------
    marginals : (n, K) array, rows sum to 1
    log_evidence : float, log p(all evidence) up to the per-frame constants
    """
    E = np.asarray(log_emission, dtype=np.float64)
    Lt = np.asarray(log_transition, dtype=np.float64)
    n, K = E.shape
    if Lt.shape != (K, K):
        raise ValueError(f'log_transition must be ({K}, {K})')
    li = np.full(K, -np.log(K)) if log_initial is None else np.asarray(log_initial, dtype=np.float64)
    mx = Lt.max(1, keepdims=True)
    T = np.exp(Lt - mx)                       # row-scaled transition
    la = np.zeros((n, K))
    a = li + E[0]
    c = a.max()
    la[0] = a
    logz = 0.
    alpha = np.exp(a - c)
    logz += c + np.log(alpha.sum())
    alpha /= alpha.sum()
    la[0] = np.log(np.maximum(alpha, 1e-300))
    for t in range(1, n):
        # alpha_t(j) = sum_i alpha_{t-1}(i) exp(Lt[i, j]) exp(E[t, j])
        pred = (alpha * np.exp(mx[:, 0] - mx.max())) @ T
        a = np.log(np.maximum(pred, 1e-300)) + mx.max() + E[t]
        c = a.max()
        alpha = np.exp(a - c)
        s = alpha.sum()
        logz += c + np.log(s)
        alpha /= s
        la[t] = np.log(np.maximum(alpha, 1e-300))
    lb = np.zeros((n, K))
    beta = np.ones(K)
    for t in range(n - 2, -1, -1):
        v = np.exp(E[t + 1] - E[t + 1].max()) * beta
        b = T @ v * np.exp(mx[:, 0] - mx.max())
        b /= b.sum()
        beta = b
        lb[t] = np.log(np.maximum(b, 1e-300))
    lm = la + lb
    lm -= lm.max(1, keepdims=True)
    m = np.exp(lm)
    m /= m.sum(1, keepdims=True)
    return m, float(logz)
