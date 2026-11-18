"""Gain-field PRF models (braincoder.models.gain_field)."""
import numpy as np
import pandas as pd
import keras
import pytest
from keras import ops

from braincoder.hrf import SPMHRFModel
from braincoder.models import (GaussianPRF2D, GaussianPRF2DWithHRF, DifferenceOfGaussiansPRF2D,
                               DifferenceOfGaussiansPRF2DWithHRF, GainFieldPRF2D, GainFieldPRF2DWithHRF,
                               GainFieldDoGPRF2D, GainFieldDoGPRF2DWithHRF, GainField, FieldShape,
                               GaussianField, DoGField)
from braincoder.optimize import ParameterFitter
from braincoder.stimuli import make_gain_field_paradigm

PRF_LABELS = ['x', 'y', 'sd', 'baseline', 'amplitude']


def _grid(n=21, radius=5.):
    x = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(x, x)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


GRID = _grid()
# Eight locations on a ring of 4 deg (an arbitrary choice for the tests).
ANGLES = np.arange(8) * np.pi / 4
POSITIONS = (4. * np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)).astype(np.float32)


def _items_images(rng, n_frames, positions=POSITIONS, radius=.8):
    """Discs at every location (a search display), random contrast per frame."""
    d = np.linalg.norm(GRID[None, :, :] - positions[:, None, :], axis=2)       # (L, G)
    discs = (d < radius).astype(np.float32)
    on = rng.random((n_frames, len(positions))) > .2
    return (on.astype(np.float32) @ discs).clip(0, 1)


def _roles(rng, n_frames, n_loc=8):
    """Per frame: one 'target' and one 'distractor' location (different)."""
    out = []
    for _ in range(n_frames):
        t, d = rng.choice(n_loc, 2, replace=False)
        out.append({'target': int(t), 'distractor': int(d)})
    return out


def _prfs(n, rng, ecc=4.):
    angle = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(.5, 1.2, n) * ecc
    return pd.DataFrame({'x': r * np.cos(angle), 'y': r * np.sin(angle),
                         'sd': rng.uniform(.8, 2., n), 'baseline': rng.normal(0, .1, n),
                         'amplitude': rng.uniform(.5, 1.5, n)}).astype(np.float32)


# --------------------------------------------------------------------- layout

def test_paradigm_helper_layout():
    images = np.arange(3 * len(GRID), dtype=np.float32).reshape(3, len(GRID))
    pos = np.array([[0., 0.], [1., 2.], [-3., 4.]], dtype=np.float32)
    assignments = [{'a': 1},                         # index
                   {'a': (-3., 4.), 'b': [0, 1]},    # (x, y) tuple, list of locations
                   {'b': {2: .5}}]                   # {location: weight}
    par = make_gain_field_paradigm(images, assignments, pos, ['a', 'b'], grid_coordinates=GRID)

    assert par.shape == (3, len(GRID) + 2 * 3)
    assert par.columns.names == ['kind', 'x | gain', 'y | location']
    assert par.columns[0] == ('image', float(GRID[0, 0]), float(GRID[0, 1]))
    np.testing.assert_array_equal(par['image'].values, images)
    field = par['field']
    assert list(field.columns) == [('a', 0), ('a', 1), ('a', 2), ('b', 0), ('b', 1), ('b', 2)]
    np.testing.assert_array_equal(field.values, [[0, 1, 0, 0, 0, 0],
                                                 [0, 0, 1, 1, 1, 0],
                                                 [0, 0, 0, 0, 0, .5]])
    # The (T, L, K) array form gives the same paradigm.
    W = np.transpose(field.values.reshape(3, 2, 3), (0, 2, 1))
    pd.testing.assert_frame_equal(
        par, make_gain_field_paradigm(images, W, pos, ['a', 'b'], grid_coordinates=GRID))

    model = GainFieldPRF2D(GRID, pos, ['a', 'b'])
    pd.testing.assert_frame_equal(model.make_paradigm(images, assignments), par)
    with pytest.raises(ValueError, match='not one of field_positions'):
        make_gain_field_paradigm(images, [{'a': (9., 9.)}] * 3, pos, ['a', 'b'])
    with pytest.raises(ValueError, match='columns'):
        model.predict(images, None)


def test_parameter_labels_are_generated_and_unique():
    model = GainFieldPRF2D(GRID, POSITIONS, ['attended', 'ignored'])
    assert model.parameter_labels == PRF_LABELS + ['field_sigma', 'field_gain_attended',
                                                   'field_gain_ignored']
    model = GainFieldDoGPRF2D(GRID, POSITIONS, ['a', 'b', 'c'],
                              fields=[GainField('slow', GaussianField(), ['a', 'b']),
                                      GainField('fast', DoGField(), ['c'], clamped=False)],
                              image_scales={'scale': np.ones(len(GRID))})
    assert model.parameter_labels == (
        ['x', 'y', 'sd', 'baseline', 'amplitude', 'srf_amplitude', 'srf_size',
         'slow_sigma', 'slow_gain_a', 'slow_gain_b',
         'fast_sigma', 'fast_surround_ratio', 'fast_surround_weight', 'fast_gain_c', 'scale'])
    with pytest.raises(ValueError, match='not unique'):
        GainFieldPRF2D(GRID, POSITIONS, ['a'], fields=[GainField('f'), GainField('f')])
    with pytest.raises(ValueError, match='unknown gain labels'):
        GainFieldPRF2D(GRID, POSITIONS, ['a'], fields=[GainField('f', gain_labels=['zzz'])])


# ----------------------------------------------------------- null model + sign

@pytest.mark.parametrize('kind', ['gauss', 'dog', 'gauss_hrf', 'dog_hrf'])
def test_null_field_equals_the_prf_model_underneath(kind):
    rng = np.random.default_rng(0)
    images = _items_images(rng, 40)
    hrf = SPMHRFModel(tr=1.5)
    fields = [GainField('slow'), GainField('fast', DoGField(), clamped=False)]
    kw = dict(fields=fields, image_scales={'items': np.ones(len(GRID))})
    if kind == 'gauss':
        model, parent = GainFieldPRF2D(GRID, POSITIONS, ['target', 'distractor'], **kw), GaussianPRF2D(GRID)
    elif kind == 'dog':
        model, parent = GainFieldDoGPRF2D(GRID, POSITIONS, ['target', 'distractor'], **kw), \
            DifferenceOfGaussiansPRF2D(GRID)
    elif kind == 'gauss_hrf':
        model = GainFieldPRF2DWithHRF(GRID, POSITIONS, ['target', 'distractor'], hrf_model=hrf, **kw)
        parent = GaussianPRF2DWithHRF(GRID, hrf_model=hrf)
    else:
        model = GainFieldDoGPRF2DWithHRF(GRID, POSITIONS, ['target', 'distractor'], hrf_model=hrf, **kw)
        parent = DifferenceOfGaussiansPRF2DWithHRF(GRID, hrf_model=hrf)

    prf = _prfs(6, rng)
    if 'dog' in kind:
        prf['srf_amplitude'], prf['srf_size'] = np.float32(.3), np.float32(2.)
    pars = model.add_field_parameters(prf, slow_sigma=1., fast_sigma=1.5, fast_surround_ratio=2.,
                                      fast_surround_weight=.5)
    paradigm = model.make_paradigm(images, _roles(rng, 40))
    np.testing.assert_allclose(model.predict(paradigm, pars).values,
                               parent.predict(images, prf).values, atol=1e-5, rtol=1e-5)


def test_positive_gain_raises_the_response_at_its_location():
    rng = np.random.default_rng(1)
    images = _items_images(rng, 20)
    model = GainFieldPRF2D(GRID, POSITIONS, ['target'])
    prf = pd.DataFrame({'x': POSITIONS[:1, 0], 'y': POSITIONS[:1, 1], 'sd': [1.], 'baseline': [0.],
                        'amplitude': [1.]})
    paradigm = model.make_paradigm(images, [{'target': 0}] * 20)
    null = model.predict(paradigm, model.add_field_parameters(prf, field_sigma=1.))
    up = model.predict(paradigm, model.add_field_parameters(prf, field_sigma=1., field_gain_target=.5))
    down = model.predict(paradigm, model.add_field_parameters(prf, field_sigma=1., field_gain_target=-.5))
    on = images @ (np.linalg.norm(GRID - POSITIONS[0], axis=1) < 1) > 0
    assert on.any()
    assert (up.values[on] > null.values[on] + 1e-4).all()
    assert (down.values[on] < null.values[on] - 1e-4).all()


def test_clamp_keeps_the_modulation_non_negative():
    rng = np.random.default_rng(2)
    images = _items_images(rng, 10)
    model = GainFieldPRF2D(GRID, POSITIONS, ['target'])
    paradigm = model.make_paradigm(images, [{'target': 0}] * 10)
    pars = model.add_field_parameters(_prfs(1, rng), field_sigma=1., field_gain_target=-5.)
    assert (model.get_modulated_images(paradigm, pars).values >= 0).all()
    unclamped = GainFieldPRF2D(GRID, POSITIONS, ['target'], clamp=False)
    assert (unclamped.get_modulated_images(paradigm, pars).values < 0).any()


# ---------------------------------------------- two fields, against numpy

class AnnulusField(FieldShape):
    """The example from the FieldShape docstring."""
    parameter_labels = ['radius', 'width']
    transformations = ['softplus', 'softplus']

    def profile(self, dx, dy, radius, width):
        r = ops.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        return ops.exp(-(r - radius) ** 2 / (2 * width ** 2))


def test_two_fields_of_different_shapes_match_a_numpy_reference():
    rng = np.random.default_rng(3)
    T = 25
    images = _items_images(rng, T)
    mask = (np.linalg.norm(GRID, axis=1) > 3).astype(np.float32)
    model = GainFieldPRF2D(
        GRID, POSITIONS, ['sustained', 'onset'],
        fields=[GainField('slow', GaussianField(), ['sustained']),
                GainField('fast', DoGField(), ['onset'], clamped=False),
                GainField('ring', AnnulusField(), ['onset'])],
        image_scales={'items': mask})
    W = (rng.random((T, 8, 2)) > .6) * rng.random((T, 8, 2))
    paradigm = model.make_paradigm(images, W.astype(np.float32))
    prf = _prfs(5, rng)
    vals = dict(slow_sigma=1.2, slow_gain_sustained=-.7, fast_sigma=.8, fast_surround_ratio=2.5,
                fast_surround_weight=.4, fast_gain_onset=.6, ring_radius=1., ring_width=.5,
                ring_gain_onset=-.9, items=.6)
    pars = model.add_field_parameters(prf, **vals)

    d2 = ((GRID[None, :, :] - POSITIONS[:, None, :]) ** 2).sum(-1)                 # (L, G)
    gauss = np.exp(-d2 / (2 * 1.2 ** 2))
    dog = np.exp(-d2 / (2 * .8 ** 2)) - .4 * np.exp(-d2 / (2 * (2.5 * .8) ** 2))
    ring = np.exp(-(np.sqrt(d2) - 1.) ** 2 / (2 * .5 ** 2))
    F_slow = -.7 * W[:, :, 0] @ gauss
    F_fast = .6 * W[:, :, 1] @ dog
    F_ring = -.9 * W[:, :, 1] @ ring
    S = images * (1 + (.6 - 1) * mask)
    S_eff = S * (np.maximum(1 + F_slow + F_ring, 0) + F_fast)
    expected = GaussianPRF2D(GRID).predict(S_eff, prf)

    np.testing.assert_allclose(model.get_modulated_images(paradigm, pars).values, S_eff, atol=1e-5)
    np.testing.assert_allclose(model.predict(paradigm, pars).values, expected.values, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------- transforms, ties

def test_parameter_transforms_round_trip():
    hrf = SPMHRFModel(tr=1.5)
    model = GainFieldDoGPRF2DWithHRF(
        GRID, POSITIONS, ['a', 'b'], hrf_model=hrf, flexible_hrf_parameters=True,
        fields=[GainField('pos', gains='positive'), GainField('neg', DoGField(), gains='negative'),
                GainField('free', gain_labels=['a'])],
        image_scales={'s': np.ones(len(GRID))})
    rng = np.random.default_rng(4)
    prf = _prfs(4, rng)
    prf['srf_amplitude'], prf['srf_size'] = np.float32(.3), np.float32(2.)
    pars = model.add_field_parameters(prf, pos_sigma=1., pos_gain_a=.3, pos_gain_b=.2, neg_sigma=2.,
                                      neg_surround_ratio=1.5, neg_surround_weight=.3, neg_gain_a=-.4,
                                      neg_gain_b=-.1, free_sigma=.7, free_gain_a=-1.3, s=2.)
    pars['hrf_delay'], pars['hrf_dispersion'] = np.float32(6.), np.float32(1.)
    pars = pars[model.parameter_labels].values.astype(np.float32)
    raw = model._transform_parameters_backward(ops.convert_to_tensor(pars))
    back = ops.convert_to_numpy(model._transform_parameters_forward(raw))
    np.testing.assert_allclose(back, pars, rtol=1e-4, atol=1e-5)


def test_ties_hold_in_predictions_and_transforms():
    model = GainFieldPRF2D(GRID, POSITIONS, ['a', 'b', 'c'],
                           fields=[GainField('f', gain_labels=['a', 'b']), GainField('g', gain_labels=['c'])],
                           ties={'f_gain_b': {'f_gain_a': -1 / 3}, 'g_sigma': 'f_sigma', 'g_gain_c': 0.})
    assert model.field_parameter_labels(include_tied=False) == ['f_sigma', 'f_gain_a']
    rng = np.random.default_rng(5)
    pars = model.add_field_parameters(_prfs(3, rng), f_sigma=1.3, f_gain_a=.6)
    np.testing.assert_allclose(pars['f_gain_b'], -.2, rtol=1e-6)
    np.testing.assert_allclose(pars['g_sigma'], 1.3)
    assert (pars['g_gain_c'] == 0).all()
    # Whatever is in a tied raw slot, the forward transform restores the tie.
    raw = ops.convert_to_numpy(model._transform_parameters_backward(
        ops.convert_to_tensor(pars[model.parameter_labels].values)))
    raw[:, model.parameter_labels.index('f_gain_b')] = 5.
    native = ops.convert_to_numpy(model._transform_parameters_forward(ops.convert_to_tensor(raw)))
    np.testing.assert_allclose(native, pars[model.parameter_labels].values, rtol=1e-4, atol=1e-5)
    # A wrong value in a tied column of a DataFrame is overwritten too.
    images = _items_images(rng, 10)
    paradigm = model.make_paradigm(images, [{'a': 0, 'b': 1, 'c': 2}] * 10)
    wrong = pars.copy()
    wrong['f_gain_b'] = np.float32(3.)
    pd.testing.assert_frame_equal(model.predict(paradigm, wrong), model.predict(paradigm, pars))


def test_field_parameters_must_be_identical_across_voxels():
    model = GainFieldPRF2D(GRID, POSITIONS, ['a'])
    pars = model.add_field_parameters(_prfs(3, np.random.default_rng(6)), field_sigma=1.)
    pars.loc[1, 'field_gain_a'] = .5
    paradigm = model.make_paradigm(np.zeros((2, len(GRID))), [{'a': 0}] * 2)
    with pytest.raises(ValueError, match='differ between voxels'):
        model.predict(paradigm, pars)


# -------------------------------------------------------------------- recovery

def _recover(model, paradigm, prf, truth, init, noise=.05, **fit_kwargs):
    pars = model.add_field_parameters(prf, **truth)
    keras.utils.set_random_seed(0)
    data = model.simulate(paradigm, pars, noise=noise)
    data.index = paradigm.index
    fitter = ParameterFitter(model, data, paradigm)
    init_pars = model.add_field_parameters(prf, **init)
    return fitter.fit(init_pars=init_pars, fixed_pars=model.prf_parameter_labels(),
                      shared_pars=model.field_parameter_labels(include_tied=False),
                      progressbar=False, **fit_kwargs)


def test_recover_shared_gains_single_trial_betas():
    """Single-trial betas: 8 items, a target and a distractor per trial."""
    rng = np.random.default_rng(7)
    n_trials = 240
    model = GainFieldPRF2D(GRID, POSITIONS, ['target', 'distractor'])
    paradigm = model.make_paradigm(_items_images(rng, n_trials), _roles(rng, n_trials))
    truth = dict(field_sigma=1.5, field_gain_target=.3, field_gain_distractor=-.2)
    fit = _recover(model, paradigm, _prfs(60, rng), truth,
                   dict(field_sigma=1., field_gain_target=0., field_gain_distractor=0.),
                   max_n_iterations=1500, learning_rate=.02, r2_atol=1e-7)
    for label, value in truth.items():
        assert fit[label].iloc[0] == pytest.approx(value, abs=.05), label


def test_recover_shared_gains_time_courses_with_hrf():
    """BOLD time courses: a sustained zero-sum field plus a transient one."""
    rng = np.random.default_rng(8)
    T = 300
    pos = POSITIONS[::2]                                                    # 4 locations
    hrf = SPMHRFModel(tr=1.5)
    model = GainFieldPRF2DWithHRF(
        GRID, pos, ['attended', 'unattended', 'onset'], hrf_model=hrf,
        fields=[GainField('sustained', gain_labels=['attended', 'unattended']),
                GainField('transient', gain_labels=['onset'], clamped=False)],
        ties={'sustained_gain_unattended': {'sustained_gain_attended': -1 / 3},
              'transient_sigma': 'sustained_sigma'})
    # A bar sweeping across the field + items at the 4 locations.
    images = _items_images(rng, T, positions=pos)
    bar_x = np.linspace(-5, 5, 30)
    for t in range(T):
        images[t] = np.maximum(images[t], np.abs(GRID[:, 0] - bar_x[t % 30]) < .6)
    attended = np.repeat(rng.integers(4, size=T // 50), 50)               # blocks
    onsets = rng.random((T, 4)) < .1
    W = np.zeros((T, 4, 3), np.float32)
    W[np.arange(T), attended, 0] = 1
    W[:, :, 1] = 1 - W[:, :, 0]
    W[:, :, 2] = onsets
    paradigm = model.make_paradigm(images, W)
    truth = dict(sustained_sigma=1.5, sustained_gain_attended=.3, transient_gain_onset=-.4)
    fit = _recover(model, paradigm, _prfs(60, rng), truth,
                   dict(sustained_sigma=1., sustained_gain_attended=0., transient_gain_onset=0.),
                   noise=.005,   # the BOLD signal here has an SD of about .03
                   max_n_iterations=1500, learning_rate=.02, r2_atol=1e-7)
    for label, value in truth.items():
        assert fit[label].iloc[0] == pytest.approx(value, abs=.06), label
    assert fit['sustained_gain_unattended'].iloc[0] == pytest.approx(
        -fit['sustained_gain_attended'].iloc[0] / 3, abs=1e-5)
