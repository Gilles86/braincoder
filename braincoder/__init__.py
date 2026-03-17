"""Top-level package for braincoder."""

__author__ = """Gilles de Hollander"""
__email__ = 'giles.de.hollander@gmail.com'
__version__ = '0.2.0'

from .models import (
    EncodingModel, EncodingRegressionModel, HRFEncodingModel,
    GaussianPRF, RegressionGaussianPRF, VonMisesPRF, LogGaussianPRF,
    GaussianPRFWithHRF, LogGaussianPRFWithHRF, AlphaGaussianPRF,
    RegressionAlphaGaussianPRF, GaussianPRFOnGaussianSignal,
    GaussianPointPRF2D, GaussianMixturePRF2D, GaussianPRF2D,
    GaussianPRF2DAngle, GaussianPRF2DWithHRF, GaussianPRF2DAngleWithHRF,
    DifferenceOfGaussiansPRF2D, DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2D, DivisiveNormalizationGaussianPRF2DWithHRF,
    DiscreteModel, LinearModel, LinearModelWithBaseline, LinearModelWithBaselineHRF,
)
from .optimize import WeightFitter, ParameterFitter, ResidualFitter
try:
    from .optimize import (
        StimulusFitter, CustomStimulusFitter,
        SzinteStimulus, SzinteStimulus2, make_aperture_stimuli,
    )
except ImportError:
    pass