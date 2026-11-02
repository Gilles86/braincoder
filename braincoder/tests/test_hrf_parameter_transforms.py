import numpy as np
import pytest

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    GaussianPRF2DWithHRF,
    DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2DWithHRF,
)

GRID = np.stack(np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10)), -1).reshape(-1, 2).astype(np.float32)
VALUES = dict(x=1., y=-.5, sd=1.2, baseline=.3, amplitude=2., rf_amplitude=2., srf_amplitude=.4,
              srf_size=2.5, neural_baseline=.6, surround_baseline=1.3, bold_baseline=.2,
              hrf_delay=5.3, hrf_dispersion=1.1)


@pytest.mark.parametrize('cls', [GaussianPRF2DWithHRF, DifferenceOfGaussiansPRF2DWithHRF,
                                 DivisiveNormalizationGaussianPRF2DWithHRF])
@pytest.mark.parametrize('flexible', [False, True])
def test_transform_round_trip(cls, flexible):
    model = cls(grid_coordinates=GRID, paradigm=np.zeros((5, len(GRID)), np.float32),
                hrf_model=SPMHRFModel(tr=1.6), flexible_hrf_parameters=flexible)
    pars = np.array([[VALUES[k] for k in model.parameter_labels]], np.float32)
    back = model._transform_parameters_forward(model._transform_parameters_backward(pars)).numpy()
    np.testing.assert_allclose(back, pars, rtol=1e-4, atol=1e-4)
