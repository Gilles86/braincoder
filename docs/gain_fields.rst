==============================
Gain fields (attention fields)
==============================

You have stimulus images on a pixel grid, and on every frame (a TR or a
trial) some locations are special: attended, salient, rewarded, expected,
where the target or a distractor is. The question is whether the stimulus
near those locations drives visual cortex more or less than the same
stimulus elsewhere. A *gain-field* PRF model answers it on the stimulus
grid: a field around each special location multiplies the image before
every voxel's ordinary PRF pools it.

.. math::

    F_t(g) &= \sum_{\text{fields}} \sum_k \sum_\ell W_{t\ell k}\, \gamma_{k}\, A(g; \ell) \\
    M_t(g) &= \max\bigl(1 + F_t(g),\, 0\bigr) \\
    r_v(t) &= \text{baseline}_v + \text{amplitude}_v \sum_g S_t(g)\, M_t(g)\, \mathrm{RF}_v(g)

- :math:`W_{t\ell k}` (in the paradigm): weight of gain type :math:`k` at
  location :math:`\ell` on frame :math:`t`, usually 0 or 1.
- :math:`\gamma_k`: one gain per gain type, signed (> 0 enhances,
  < 0 suppresses). Shared by all voxels.
- :math:`A(g;\ell)`: the field shape centred on location :math:`\ell`, peak 1
  (a Gaussian with a shared width ``sigma`` by default).
- :math:`\mathrm{RF}_v`: the voxel's receptive field from the model underneath
  (``GaussianPRF2D``, ``DifferenceOfGaussiansPRF2D``, ...), with the same
  amplitude convention.

With every gain at 0, the model predicts exactly what the PRF model
underneath predicts. So you can fit the PRFs first and then ask how much the
gain fields add.

**Tutorial:** :doc:`auto_examples/02_gain_fields/01_gain_fields` simulates a
population, fits shared gains with the PRFs fixed, combines two fields and adds
a field shape of your own (also :ref:`tutorial_lesson8`).

Step 1: the paradigm
====================

The paradigm holds the images *and* the per-frame weights.
:func:`braincoder.stimuli.make_gain_field_paradigm` (or ``model.make_paradigm``)
builds it:

.. code-block:: python

    import numpy as np
    from braincoder.models import GainFieldPRF2D

    # Pixel grid: (n_grid, 2) x and y in degrees.
    x = np.linspace(-6, 6, 41)
    gx, gy = np.meshgrid(x, x)
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)

    # Every location a field can sit on: any number, anywhere.
    angles = np.arange(8) * np.pi / 4
    field_positions = 4 * np.stack([np.cos(angles), np.sin(angles)], 1)    # 8 items at 4 deg

    model = GainFieldPRF2D(grid, field_positions, gain_labels=['target', 'distractor'])
    model.parameter_labels
    # ['x', 'y', 'sd', 'baseline', 'amplitude',
    #  'field_sigma', 'field_gain_target', 'field_gain_distractor']

    # images: (n_trials, n_grid) or (n_trials, n_x, n_y), flattened like grid.
    # One dict per trial: gain label -> location index, (x, y) tuple,
    # list of locations, or {location: weight}.
    assignments = [{'target': 2, 'distractor': 5},
                   {'target': 0},                       # no distractor on this trial
                   ...]
    paradigm = model.make_paradigm(images, assignments, index=betas.index)

Columns: ``('image', x, y)`` for every pixel, then
``('field', gain_label, location_index)``. An array
``W`` of shape ``(n_frames, n_locations, n_gains)`` works instead of the list
of dicts, which is handy for fractional weights (e.g. the fraction of a TR a
distractor was on). If the special things move freely, pass every position
that occurs as ``field_positions``.

Step 2: fit
===========

The field parameters (widths and gains) belong to the model, not to a voxel.
The parameters DataFrame repeats them on every row, and they must be equal
across rows. Fit them with ``shared_pars``. The robust route is to fit the PRFs
first with the model underneath, then fit the shared terms with the PRFs held
fixed:

.. code-block:: python

    from braincoder.models import GaussianPRF2D
    from braincoder.optimize import ParameterFitter

    # (a) PRFs from the mapping data, as usual, with GaussianPRF2D.
    prf_pars = ...                                   # columns x, y, sd, baseline, amplitude

    # (b) Gain fields with the PRFs fixed. Gains start at 0 (= the PRF model).
    init = model.add_field_parameters(prf_pars, field_sigma=1.0)
    fitter = ParameterFitter(model, betas, paradigm)
    pars = fitter.fit(init_pars=init,
                      fixed_pars=model.prf_parameter_labels(),
                      shared_pars=model.field_parameter_labels(include_tied=False),
                      learning_rate=0.02, max_n_iterations=2000, r2_atol=1e-7)
    pars.iloc[0][model.field_parameter_labels()]

Leave out ``fixed_pars`` to refine PRFs and gains jointly. Check recovery on
simulated data at your noise level (``model.simulate(paradigm, pars,
noise=...)``) before trusting a gain: gains and widths trade off when the
field is much wider or narrower than the items.

Time courses with an HRF
========================

``GainFieldPRF2DWithHRF`` (and ``GainFieldDoGPRF2DWithHRF``) apply the
modulation to the images frame by frame, then the HRF. Several fields can
act at once. Below, a sustained field marks the location attended in each
block, with a zero-sum constraint (the gain at the attended location and the
gain at the 3 others sum to 0, so a uniform change of the whole display is
left to the PRFs), and a transient field, which shares its width, responds to
brief onsets. The transient field sits outside the clip at 0. A third term
scales the drive of the items against a mapping bar, since both are painted
as 1 in the images:

.. code-block:: python

    from braincoder.hrf import SPMHRFModel
    from braincoder.models import GainFieldDoGPRF2DWithHRF, GainField

    positions = 4 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) / np.sqrt(2)
    item_mask = np.hypot(grid[:, 0], grid[:, 1]) > item_radius           # your geometry

    model = GainFieldDoGPRF2DWithHRF(
        grid, positions, ['attended', 'unattended', 'onset'],
        hrf_model=SPMHRFModel(tr=1.5),
        fields=[GainField('sustained', gain_labels=['attended', 'unattended']),
                GainField('transient', gain_labels=['onset'], clamped=False)],
        image_scales={'item_scale': item_mask},
        ties={'sustained_gain_unattended': {'sustained_gain_attended': -1 / 3},
              'transient_sigma': 'sustained_sigma'})

    # W: (n_TRs, 4, 3). Column 0: 1 at the attended location of the block;
    # column 1: 1 elsewhere; column 2: fraction of the TR an onset was on.
    paradigm = model.make_paradigm(images, W, index=bold.index)

``ties`` makes a parameter a function of others: a number (a constant, e.g.
``0.`` to switch a gain off), another label (equal), or ``{label: coef}`` (a
linear combination). Tied parameters are filled in by the model, so
predictions and fitted values always satisfy the constraint, and you leave
them out of ``shared_pars`` (``field_parameter_labels(include_tied=False)``).
``image_scales`` adds a positive model-level parameter per mask; its null
value is 1.

Extending
=========

**Several fields.** ``fields`` is a list of :class:`~braincoder.models.GainField`
(name, shape, gain labels, sign constraint, inside or outside the clip). The
parameters are named ``f'{name}_{shape_parameter}'`` and
``f'{name}_gain_{label}'``.

**A new field shape** is a subclass of :class:`~braincoder.models.FieldShape`
with parameter labels, their transforms (unconstrained → native) and one
method, written with ``keras.ops`` so it runs on every backend:

.. code-block:: python

    from keras import ops
    from braincoder.models import FieldShape, GainField, GainFieldPRF2D

    class AnnulusField(FieldShape):
        parameter_labels = ['radius', 'width']
        transformations = ['softplus', 'softplus']

        def profile(self, dx, dy, radius, width):
            # dx, dy: (1, n_locations, n_grid); parameters: (n_batches, 1, 1)
            r = ops.sqrt(dx ** 2 + dy ** 2 + 1e-12)
            return ops.exp(-(r - radius) ** 2 / (2 * width ** 2))

    model = GainFieldPRF2D(grid, positions, ['attended'],
                           fields=[GainField('ring', AnnulusField())])
    # parameters ... 'ring_radius', 'ring_width', 'ring_gain_attended'

``GaussianField`` and ``DoGField`` (centre minus a wider surround) ship with
braincoder.

**Another combination rule.** Pass ``combination=f`` with ``f(images,
field_sums) -> images`` (one field sum per field, in order), e.g. divisive:
``lambda S, F: S / (1 + ops.softplus(sum(F)))``; or subclass and override
``_combine``.

**Another PRF kernel.** Combine :class:`~braincoder.models.GainFieldMixin`
with any image PRF model:

.. code-block:: python

    from braincoder.models import GainFieldMixin, DivisiveNormalizationGaussianPRF2D

    class GainFieldDNPRF2D(GainFieldMixin, DivisiveNormalizationGaussianPRF2D):
        pass

Limitations
===========

- Field parameters are model-level (one value for all voxels in the
  parameters DataFrame). Fit different areas as separate models if their
  fields may differ.
- Decoding (``get_stimulus_pdf``) is not meaningful with these models: decode
  with the PRF model underneath.
- ``ParameterFitter`` has no bounds; constrain gains with
  ``GainField(gains='positive' | 'negative')`` and widths with
  ``GaussianField(sigma_min=...)``.
- Memory: the modulated images are ``n_frames x n_grid`` (shared fields
  never build a voxels x frames x pixels tensor), so the cost is close to the
  PRF model underneath.

API
===

.. autoclass:: braincoder.models.GainFieldPRF2D
   :show-inheritance:

.. autoclass:: braincoder.models.GainFieldMixin
   :members: field_parameter_labels, prf_parameter_labels, add_field_parameters, apply_ties, make_paradigm, get_modulated_images

.. autoclass:: braincoder.models.GainField

.. autoclass:: braincoder.models.FieldShape
   :members: profile

.. autoclass:: braincoder.models.GaussianField

.. autoclass:: braincoder.models.DoGField

.. autofunction:: braincoder.stimuli.make_gain_field_paradigm
