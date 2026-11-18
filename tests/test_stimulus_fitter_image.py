"""StimulusFitter on images: free and masked decoding, with the norms applied to one tensor."""
import keras
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(keras.backend.backend() != 'tensorflow', reason='StimulusFitter needs TensorFlow')


def _problem(n_trials=6, n_voxels=40, res=12):
    from braincoder.models import GaussianPRF2D
    rng = np.random.default_rng(0)
    ax = np.linspace(-5, 5, res, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    grid = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()}).astype(np.float32)
    images = np.zeros((n_trials, res, res), np.float32)
    for t in range(n_trials):
        images[t, :, t * 2:t * 2 + 2] = 1.
    paradigm = pd.DataFrame(images.reshape(n_trials, -1), index=pd.Index(range(n_trials), name='frame'))
    pars = pd.DataFrame({'x': rng.uniform(-4, 4, n_voxels), 'y': rng.uniform(-4, 4, n_voxels),
                         'sd': rng.uniform(1, 2, n_voxels), 'baseline': 0., 'amplitude': 1.}).astype(np.float32)
    model = GaussianPRF2D(grid_coordinates=grid)
    data = pd.DataFrame(np.asarray(model.predict(paradigm, pars)), index=paradigm.index).astype(np.float32)
    return model, data, pars, images, np.eye(n_voxels, dtype=np.float32) * .1


@pytest.mark.parametrize('masked', [False, True])
def test_image_decoding_shape_mask_and_l2(masked):
    from braincoder.optimize import StimulusFitter
    model, data, pars, images, omega = _problem()
    mask = images > 0 if masked else None

    def decode(l2):
        fitter = StimulusFitter(model=model, data=data, omega=omega, parameters=pars)
        return fitter.fit(l2_norm=l2, learning_rate=.05, max_n_iterations=60, min_n_iterations=100,
                          progressbar=False, mask=mask).to_numpy()

    weak, strong = decode(1e-3), decode(10.)
    assert weak.shape == (images.shape[0], images[0].size)
    assert np.isfinite(weak).all()
    if masked:
        assert (weak[~mask.reshape(len(images), -1)] <= 1e-5).all()      # pixels outside the mask stay at the empty value
    assert (strong ** 2).sum() < (weak ** 2).sum()
