#!/usr/bin/env python
"""Re-test the behavioural claims in the braincoder skill against the installed version.

    KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES= python check_skill_claims.py

About a minute on CPU. Each line reads OK (the skill is still right) or CHANGED (the
library moved: update SKILL.md, or the trap has been fixed and the rule can go).
Exits 1 if anything changed. Most checks document traps, so a CHANGED is often good news.
"""
import logging
import os
import sys
import warnings

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import pandas as pd
import keras
from keras import ops
import braincoder
from braincoder.hrf import SPMHRFModel
from braincoder.models import (EncodingModel, GaussianPRF, GaussianPRF2D, GaussianPRF2DWithHRF,
                               GaussianPRFWithHRF, HRFEncodingModel)
from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter
from braincoder.stimuli import Stimulus
from braincoder.utils import get_rsq

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
print(f'braincoder {braincoder.__version__} from {braincoder.__file__}; backend {keras.backend.backend()}')

rng = np.random.default_rng(0)
V, N = 60, 240
TRUE = pd.DataFrame({'mu': rng.uniform(5, 35, V), 'sd': rng.uniform(4, 10, V),
                     'amplitude': rng.uniform(.5, 2, V), 'baseline': rng.normal(0, .3, V)},
                    index=pd.Index([f'v{i}' for i in range(V)], name='voxel')).astype(np.float32)
X = rng.uniform(5, 35, N).astype(np.float32)
PARADIGM = pd.DataFrame({'x': X}, index=pd.Index(np.arange(N), name='trial'))
DATA = GaussianPRF().simulate(PARADIGM, TRUE, noise=.5)
DATA.index = PARADIGM.index
RUNS = np.repeat(np.arange(8), 30)
TR, TE = RUNS < 6, RUNS >= 6
GRID = np.linspace(0, 40, 81).astype(np.float32)
SEARCH = dict(mu=np.linspace(5, 35, 16), sd=np.linspace(3, 15, 7), amplitude=[1.], baseline=[0.])
changed = []


def claim(text, fn):
    try:
        ok = bool(fn())
    except Exception as e:
        ok = False
        text += f'  [{type(e).__name__}: {str(e)[:90]}]'
    print(('OK      ' if ok else 'CHANGED ') + text, flush=True)
    if not ok:
        changed.append(text)


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


# ---- the canonical recipe -------------------------------------------------
fitter = ParameterFitter(GaussianPRF(), DATA[TR], PARADIGM[TR])
grid_pars = fitter.fit_grid(**SEARCH, use_correlation_cost=True)
init = fitter.refine_baseline_and_amplitude(grid_pars)
PARS = fitter.fit(init_pars=init, max_n_iterations=500, min_n_iterations=500,
                  noise_model='gaussian', progressbar=False)
model = GaussianPRF()
claim('Recipe: grid -> refine -> fit recovers mu (r > .95)', lambda: np.corrcoef(PARS.mu, TRUE.mu)[0, 1] > .95)
claim('Rule 5: raw grid R2 is ~0, refine fixes it',
      lambda: fitter.get_rsq(grid_pars).mean() < .1 < .3 < fitter.get_rsq(init).mean())


def zero_amplitude_grid_is_useless():
    """TF raises 'not invertible'; JAX returns without error, with a uniform grid result."""
    g = fitter.fit_grid(**{**SEARCH, 'amplitude': [0.]}, use_correlation_cost=True)
    if g['mu'].nunique() != 1:
        return False
    try:
        fitter.refine_baseline_and_amplitude(g)
    except Exception:
        pass
    return True


claim('Rule 5: a grid with amplitude 0 cannot locate any voxel', zero_amplitude_grid_is_useless)
claim('Rule 10: models hold no fitted values or data', lambda: not any(
    hasattr(fitter.model, name) for name in ('parameters', 'weights', 'omega', 'dof', 'paradigm', 'data')))
claim('Rule 10: constructors refuse data and fitted values', lambda: raises(TypeError, lambda: GaussianPRF(parameters=TRUE)))
claim('Rule 10: leaving parameters out raises', lambda: raises(ValueError, lambda: GaussianPRF().predict(PARADIGM)))
claim('cvR2 via get_rsq on held-out data is sane', lambda: .2 < get_rsq(DATA[TE], model.predict(PARADIGM[TE], PARS)).mean() < .6)

# ---- rule 3: alignment ----------------------------------------------------
shifted = DATA[TR].set_axis(np.arange(1000, 1000 + TR.sum()), axis=0)
claim('Rule 3: get_rsq refuses frames without matching labels',
      lambda: raises(ValueError, lambda: get_rsq(shifted, model.predict(PARADIGM[TR], PARS))))
off_by_one = DATA.set_axis(np.arange(1, N + 1), axis=0)


def partial():
    f = ParameterFitter(GaussianPRF(), off_by_one, X[:, None])
    g = f.refine_baseline_and_amplitude(f.fit_grid(**SEARCH, use_correlation_cost=True))
    f.fit(init_pars=g, max_n_iterations=120, progressbar=False)
    return abs(f.get_rsq().mean() - f.r2.mean()) < .02


claim('Rule 3: ParameterFitter pairs rows by position if the index differs', partial)

# ---- rule 4: masked voxels ------------------------------------------------
nan_data = DATA[TR].copy()
nan_data.iloc[3, 1] = np.nan
f = ParameterFitter(GaussianPRF(), nan_data, PARADIGM[TR])
p = f.fit(init_pars=PARS, max_n_iterations=110, progressbar=False)
claim('Rule 4: one NaN sample -> NaN parameters', lambda: p.iloc[1].isna().all())
claim('Rule 4: ... and NaN fitter.r2, with a usable mean', lambda: np.isnan(f.r2.iloc[1]) and f.r2.mean() > .2)
claim('Rule 4: decoding refuses a NaN row', lambda: raises(ValueError, lambda: GaussianPRF().get_stimulus_pdf(
    DATA[TE], GRID, parameters=p, omega=np.eye(V, dtype=np.float32) * .25)))

# ---- rule 7: scale --------------------------------------------------------
big = PARADIGM[TR] * 100
f = ParameterFitter(GaussianPRF(), DATA[TR], big)
g = f.refine_baseline_and_amplitude(f.fit_grid(mu=SEARCH['mu'] * 100, sd=SEARCH['sd'] * 100, amplitude=[1.],
                                               baseline=[0.], use_correlation_cost=True))
q = f.fit(init_pars=g, max_n_iterations=300, min_n_iterations=300, progressbar=False)
claim('Rule 7: Adam moves mu at most lr x iterations', lambda: (q.mu - g.mu).abs().max() <= 3.05)

# ---- rules 8-10: decoding -------------------------------------------------
stim = np.zeros((300, 1), np.float32)
stim[::15, 0] = rng.uniform(10, 30, 20)
hrf_pars = TRUE.iloc[:20]
hrf_model = GaussianPRFWithHRF(hrf_model=SPMHRFModel(tr=1.6))
plain = GaussianPRF()


def hrf_twin():
    a_ = hrf_model.predict_stimulus_grid(GRID, parameters=hrf_pars)
    return np.allclose(a_, plain.predict_stimulus_grid(GRID, parameters=hrf_pars), atol=1e-5)


claim('Rule 8: a *WithHRF model decodes like its plain twin', hrf_twin)
claim('Rule 8: *WithHRF refuses get_expected_uncertainty', lambda: raises(
    NotImplementedError, lambda: hrf_model.get_expected_uncertainty(GRID, omega=np.eye(20))))

rf_model = GaussianPRF()
claim('ResidualFitter without weights needs stimulus_range', lambda: raises(ValueError, lambda: ResidualFitter(
    rf_model, DATA[TR], PARADIGM[TR], parameters=PARS).fit(max_n_iterations=2, progressbar=False)))
rf = ResidualFitter(rf_model, DATA[TR], PARADIGM[TR], parameters=PARS)
OMEGA, DOF = rf.fit(stimulus_range=GRID, method='t', init_dof=10., max_n_iterations=200, progressbar=False)
claim('ResidualFitter records n_failed_steps', lambda: rf.n_failed_steps == 0)
t_pdf = rf_model.get_stimulus_pdf(DATA[TE], GRID, parameters=PARS, omega=OMEGA, dof=DOF)
gauss_pdf = rf_model.get_stimulus_pdf(DATA[TE], GRID, parameters=PARS, omega=OMEGA)
claim('Rule 9: dof=None means Gaussian, and omega is required', lambda: not np.allclose(gauss_pdf.values, t_pdf.values)
      and raises(ValueError, lambda: rf_model.get_stimulus_pdf(DATA[TE], GRID, parameters=PARS)))
claim('ResidualFitter pairs rows by position', lambda: np.isfinite(ResidualFitter(
    rf_model, shifted, PARADIGM[TR], parameters=PARS).fit(stimulus_range=GRID, max_n_iterations=5, progressbar=False)[0]).all())
bad = OMEGA - np.eye(V) * (np.linalg.eigvalsh(OMEGA).min() + 1e-3)
claim('Decoding docs: get_stimulus_pdf survives non-PD omega', lambda: np.isfinite(
    rf_model.get_stimulus_pdf(DATA[TE], GRID, parameters=PARS, omega=bad.astype(np.float32)).values).all())


def low_rank():
    from braincoder.noise import LowRankNoise
    from braincoder.utils.calibration import calibration_summary
    stim_tr = rng.choice(GRID[10:71], TR.sum()).astype(np.float32)
    stim_te = rng.choice(GRID[10:71], TE.sum()).astype(np.float32)
    gen = GaussianPRF()
    train = gen.simulate(stim_tr[:, None], TRUE, noise=.5)
    test = gen.simulate(stim_te[:, None], TRUE, noise=.5)
    m = GaussianPRF()
    noise = LowRankNoise.fit(m, train, stim_tr[:, None], PARS, stimulus_range=GRID, groups=RUNS[TR])
    pdf = m.get_stimulus_pdf(test, GRID, parameters=PARS, noise_model=noise)
    return calibration_summary(pdf, stim_te)['coverage_0.95'] > .8


claim('Decoding recipe: LowRankNoise + calibration_summary', low_rank)

# ---- rule 11, 12, normalization, misc -------------------------------------
claim('Rule 11: 1D and 2D orders differ', lambda: GaussianPRF.parameter_labels[2:] == ['amplitude', 'baseline']
      and GaussianPRF2D.parameter_labels[3:] == ['baseline', 'amplitude'])
claim('Rule 12: WeightFitter has fit_intercept, off by default', lambda: 'fit_intercept' in WeightFitter.fit.__code__.co_varnames
      and WeightFitter.fit.__defaults__[-1] is False)
g2 = np.linspace(-5, 5, 20)
coords = np.stack(np.meshgrid(g2, g2), -1).reshape(-1, 2).astype(np.float32)
image = (rng.random((200, 400)) < .1).astype(np.float32)
P2 = pd.DataFrame({'x': rng.uniform(-3, 3, 10), 'y': rng.uniform(-3, 3, 10), 'sd': rng.uniform(.8, 2, 10),
                   'baseline': 0., 'amplitude': 1.}).astype(np.float32)
signal = lambda norm: GaussianPRF2DWithHRF(grid_coordinates=coords, hrf_model=SPMHRFModel(tr=1.6),
                                           rf_normalization=norm).predict(paradigm=image, parameters=P2).std().median()
claim('rf_normalization: legacy amplitude=1 signal is ~10x below peak', lambda: signal('peak') > 5 * signal('legacy'))


def transforms_round_trip():
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF, DivisiveNormalizationGaussianPRF2DWithHRF
    g = np.linspace(-5, 5, 10)
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    vals = dict(x=1., y=-.5, sd=1.2, baseline=.3, amplitude=2., rf_amplitude=2., srf_amplitude=.4, srf_size=2.5,
                neural_baseline=.6, surround_baseline=1.3, bold_baseline=.2, hrf_delay=5.3, hrf_dispersion=1.1)
    for cls in (DifferenceOfGaussiansPRF2DWithHRF, DivisiveNormalizationGaussianPRF2DWithHRF):
        m = cls(grid_coordinates=grid, hrf_model=SPMHRFModel(tr=1.6),
                flexible_hrf_parameters=True)
        p = np.array([[vals[k] for k in m.parameter_labels]], np.float32)
        back = ops.convert_to_numpy(m._transform_parameters_forward(m._transform_parameters_backward(p)))
        if not np.allclose(back, p, rtol=1e-4, atol=1e-4):
            return False
    return True


claim('Models reference: DoG and DN flexible-HRF transforms round-trip', transforms_round_trip)


def _prf_setup():
    g = np.linspace(-5, 5, 21)
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    s = (np.random.default_rng(1).random((12, len(grid))) > .6).astype(np.float32)
    return grid, s, (g[1] - g[0]) ** 2


def rf_normalization_is_reparameterisation():
    grid, s, area = _prf_setup()
    sd = np.array([.7, 1.5, 3.], np.float32)
    base = dict(x=[1., -1., 0.], y=[0., 2., -1.], sd=sd, baseline=[0., .2, -.1])
    scale = {'legacy': area / (sd * np.sqrt(2 * np.pi)), 'peak': np.ones(3), 'area': area / (2 * np.pi * sd ** 2)}
    preds = []
    for norm, w in scale.items():
        pars = pd.DataFrame({**base, 'amplitude': 2. / w}).astype(np.float32)
        m = GaussianPRF2D(grid_coordinates=grid, rf_normalization=norm)
        preds.append(np.asarray(m.predict(paradigm=s, parameters=pars[m.parameter_labels])))
    return all(np.allclose(p, preds[0], rtol=1e-3, atol=1e-4) for p in preds[1:])


claim("Models reference: 'legacy', 'peak' and 'area' predict the same after converting amplitude",
      rf_normalization_is_reparameterisation)


def plain_dn(rf_amplitude):
    """Max |prediction - (A<S,G1> + B) / (C<S,G2> + D)|, with C using |rf_amplitude|."""
    from braincoder.models import DivisiveNormalizationGaussianPRF2D
    grid, s, _ = _prf_setup()
    sd, size, srf, b, d = .8, 2.5, .1, .6, 1.3
    p = pd.DataFrame(dict(x=[1.], y=[0.], sd=[sd], rf_amplitude=[rf_amplitude], srf_amplitude=[srf], srf_size=[size],
                          neural_baseline=[b], surround_baseline=[d])).astype(np.float32)
    m = DivisiveNormalizationGaussianPRF2D(grid_coordinates=grid, rf_normalization='peak')
    got = np.asarray(m.predict(paradigm=s, parameters=p[m.parameter_labels]))[:, 0]
    r2 = ((grid - [1., 0.]) ** 2).sum(1)
    g1, g2 = s @ np.exp(-r2 / (2 * sd ** 2)), s @ np.exp(-r2 / (2 * (sd * size) ** 2))
    return np.abs(got - (rf_amplitude * g1 + b) / (srf * abs(rf_amplitude) * g2 + d)).max()


claim("Models reference: plain DN with 'peak' is (A<S,G1> + B) / (C<S,G2> + D), without -B/D",
      lambda: plain_dn(.2) < 1e-3)
claim('Models reference: a negative rf_amplitude leaves the DN denominator positive (|A| in C)',
      lambda: plain_dn(-.2) < 1e-3)

# ---- custom model template -------------------------------------------------
class SigmoidModel(EncodingModel):
    parameter_labels = ['x0', 'slope', 'amplitude', 'baseline']
    stimulus_type = Stimulus

    def __init__(self, **kwargs):
        if not hasattr(self, 'transformations'):
            self.transformations = ['identity', 'identity', 'aggressive_softplus', 'identity']
        super().__init__(**kwargs)

    def _basis_predictions(self, paradigm, parameters):
        x = paradigm[..., None, 0]
        x0, slope, amplitude, baseline = [parameters[:, None, :, i] for i in range(4)]
        z = ops.clip(-slope * (x - x0), -30., 30.)
        return baseline + amplitude / (1. + ops.exp(z))


class SigmoidModelWithHRF(HRFEncodingModel, SigmoidModel):
    def __init__(self, hrf_model=None, flexible_hrf_parameters=False, **kwargs):
        SigmoidModel.__init__(self, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)


def template():
    sig = pd.DataFrame({'x0': rng.uniform(10, 30, 20), 'slope': rng.choice([-.4, .4], 20),
                        'amplitude': 1.5, 'baseline': 0.}).astype(np.float32)
    y = SigmoidModel().simulate(PARADIGM, sig, noise=.2)
    y.index = PARADIGM.index
    f = ParameterFitter(SigmoidModel(), y, PARADIGM)
    g = f.fit_grid(x0=np.linspace(5, 35, 13), slope=[-.5, -.2, .2, .5], amplitude=[1.], baseline=[0.], use_correlation_cost=False)
    g = f.refine_baseline_and_amplitude(g)
    q = f.fit(init_pars=g, max_n_iterations=300, progressbar=False)
    h = SigmoidModelWithHRF(hrf_model=SPMHRFModel(tr=1.6))
    return np.corrcoef(q.x0, sig.x0)[0, 1] > .9 and h.predict(paradigm=stim, parameters=sig).shape == (300, 20)


claim('Custom models: the template and its HRF variant fit and predict', template)


class Extra(GaussianPRF):
    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline', 'gain']

    def _get_basis_predictions(self, model_stimulus_amplitude=False):
        return self._gain_basis

    def _gain_basis(self, paradigm, parameters):
        mu, sd, a, b, gain = [parameters[:, None, :, i] for i in range(5)]
        return b + a * gain * ops.exp(-.5 * ((paradigm[..., None, 0] - mu) / sd) ** 2)


claim('Custom models: label without transformation crashes fit but not predict', lambda: (
    Extra().predict(PARADIGM, TRUE.assign(gain=2.).astype(np.float32)).shape == (N, V)
    and raises(Exception, lambda: ParameterFitter(Extra(), DATA, PARADIGM).fit(
        init_pars=TRUE.assign(gain=2.).astype(np.float32), max_n_iterations=2, progressbar=False))))

import braincoder.hrf as hrf_module
claim('Custom models: only the two documented HRF models exist', lambda: hasattr(hrf_module, 'SPMHRFDerivativeModel')
      and not hasattr(hrf_module, 'CustomHRFModel'))

# ---- 0.7.1 fixes (CHANGED on 0.7.0 or older: see references/version_history.md) ----
claim('SPMHRFModel peaks at 5 s (kernel fix, 0.7.1; 0.7.0 peaked at 6)',
      lambda: int(np.argmax(np.asarray(SPMHRFModel(tr=1.).get_hrf()).ravel())) == 5)

print(f'\n{len(changed)} claim(s) changed.')
sys.exit(1 if changed else 0)
