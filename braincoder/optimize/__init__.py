from .weight_fitter import WeightFitter
from .parameter_fitter import ParameterFitter
from .residual_fitter import ResidualFitter
try:
    from .stimulus_fitter import (StimulusFitter, CustomStimulusFitter,
                                   SzinteStimulus, SzinteStimulus2,
                                   make_aperture_stimuli)
    _STIMULUS_FITTERS = ['StimulusFitter', 'CustomStimulusFitter', 'SzinteStimulus',
                         'SzinteStimulus2', 'make_aperture_stimuli']
except ImportError as _error:
    _STIMULUS_FITTERS = []
    _stimulus_fitter_error = _error

    def __getattr__(name):
        if name in ('StimulusFitter', 'CustomStimulusFitter', 'SzinteStimulus',
                    'SzinteStimulus2', 'make_aperture_stimuli'):
            raise ImportError(
                f'{name} needs TensorFlow and tensorflow-probability '
                f'({_stimulus_fitter_error}).')
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['WeightFitter', 'ParameterFitter', 'ResidualFitter'] + _STIMULUS_FITTERS
