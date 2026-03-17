from .weight_fitter import WeightFitter
from .parameter_fitter import ParameterFitter
from .residual_fitter import ResidualFitter
try:
    from .stimulus_fitter import (StimulusFitter, CustomStimulusFitter,
                                   SzinteStimulus, SzinteStimulus2,
                                   make_aperture_stimuli)
except ImportError:
    pass

__all__ = [
    'WeightFitter',
    'ParameterFitter',
    'ResidualFitter',
    'StimulusFitter',
    'CustomStimulusFitter',
    'SzinteStimulus',
    'SzinteStimulus2',
    'make_aperture_stimuli',
]
