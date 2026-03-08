import numpy as np
import keras
from keras import ops
from .utils import logit
from .utils.backend import softplus_inverse


def bounded_sigmoid_transform(min_val, max_val):
    """Return a (forward, backward) pair that maps reals → [min_val, max_val] via sigmoid."""
    def forward(x):
        return ops.sigmoid(x) * (max_val - min_val) + min_val
    def backward(y):
        y_scaled = (y - min_val) / (max_val - min_val)
        return ops.log(y_scaled / (1.0 - y_scaled))
    return (forward, backward)


class HRFModel(object):
    """Base class that handles HRF generation and convolution with predictions."""

    def _transform_parameters_forward(self, parameters):
        """Map unconstrained optimizer parameters to the model's native space via ``self.transformations``."""
        out = []
        for i, t in enumerate(self.transformations):
            param = parameters[:, i][:, None]
            if isinstance(t, tuple):
                out.append(t[0](param))
            elif t == 'identity':
                out.append(param)
            elif t == 'softplus':
                out.append(ops.softplus(param))
            elif t == 'sigmoid':
                out.append(ops.sigmoid(param))
            else:
                raise NotImplementedError(f"Unknown transform: {t!r}")
        return ops.concatenate(out, axis=1)

    def _transform_parameters_backward(self, parameters):
        """Inverse of :meth:`_transform_parameters_forward`."""
        out = []
        for i, t in enumerate(self.transformations):
            param = parameters[:, i][:, None]
            if isinstance(t, tuple):
                out.append(t[1](param))
            elif t == 'identity':
                out.append(param)
            elif t == 'softplus':
                out.append(softplus_inverse(param))
            elif t == 'sigmoid':
                out.append(ops.log(param / (1.0 - param)))
            else:
                raise NotImplementedError(f"Unknown transform: {t!r}")
        return ops.concatenate(out, axis=1)

    def set_unique_hrfs(self, unique_hrfs):
        self.unique_hrfs = unique_hrfs
        self._convolve = self._convolve_unique if unique_hrfs else self._convolve_shared

    def __init__(self, unique_hrfs=False):
        self.unique_hrfs = unique_hrfs
        self._convolve = self._convolve_unique if unique_hrfs else self._convolve_shared

    def convolve(self, timeseries, **kwargs):
        hrf = self.get_hrf(**kwargs)
        return self._convolve(timeseries, hrf=hrf)

    def _convolve_shared(self, timeseries, hrf):
        # timeseries: n_batch x n_timepoints x n_units
        # hrf: n_hrf_timepoints (1D) or n_hrf_timepoints x 1
        # Treat time as height, units as width: 2D conv with a (n_hrf, 1) filter
        # slides only along the time axis, leaving units unchanged.
        n_hrf = int(hrf.shape[0])
        pad_ = ops.tile(timeseries[:, :1, :], [1, n_hrf, 1])
        timeseries_padded = ops.concatenate((pad_, timeseries), axis=1)

        hrf_1d = ops.reshape(hrf, (-1,))  # ensure 1D
        # ops.conv with 4D kernel → 2D conv; reverse HRF for convolution vs cross-correlation
        kernel = ops.reshape(ops.flip(hrf_1d, axis=0), (n_hrf, 1, 1, 1))  # (fH, fW, in_C, out_C)
        cts = ops.conv(timeseries_padded[:, :, :, None], kernel,
                       strides=(1, 1), padding='valid')[:, :, :, 0]

        return cts[:, 1:, :]

    def _convolve_unique(self, timeseries, hrf):
        # timeseries: n_batch x n_timepoints x n_units
        # hrf: n_hrf_timepoints x n_units
        # 1D depthwise conv: each unit gets its own HRF filter.
        n, m = timeseries.shape[1], timeseries.shape[2]
        n_hrf = int(hrf.shape[0])

        pad_ = ops.tile(timeseries[:, :1, :], [1, n_hrf, 1])
        timeseries_padded = ops.concatenate((pad_, timeseries), axis=1)
        # timeseries_padded: (1, n+n_hrf, m) — (batch, steps, channels) for 1D depthwise conv

        # kernel: (kernel_size, channels, depth_multiplier); reverse time axis for convolution
        kernel = ops.reshape(ops.flip(hrf, axis=0), (n_hrf, m, 1))

        convolved = ops.depthwise_conv(timeseries_padded, kernel, strides=1, padding='valid')
        # convolved: (1, n+1, m) — trim leading timepoint
        return convolved[:, 1:, :]


def gamma_pdf(t, a, d, eps=1e-6):
    """Numerically stable gamma PDF."""
    from .utils.backend import lgamma
    t = ops.maximum(t, eps)
    coef = ops.power(t, a - 1) * ops.exp(-t / d)
    denom = ops.power(d, a) * ops.exp(lgamma(a))
    return coef / denom


def gamma_pdf_with_loc(t, a, d, dt, loc=0.0):
    """Gamma PDF shifted by a delay 'loc'."""
    t = ops.convert_to_tensor(t, dtype='float32')
    a = ops.convert_to_tensor(a, dtype='float32')
    d = ops.convert_to_tensor(d, dtype='float32')

    shifted_t = t - loc
    mask = shifted_t >= 0
    safe_t = ops.where(mask, shifted_t, ops.ones_like(shifted_t))
    pdf_vals = ops.where(mask, gamma_pdf(safe_t, a, d), ops.zeros_like(shifted_t))

    loc_steps = int(round(float(loc) / float(dt)))

    if loc_steps > 0:
        trimmed = pdf_vals[:-loc_steps, :]
        padding = ops.zeros([loc_steps, ops.shape(trimmed)[1]], dtype='float32')
        result = ops.concatenate([padding, trimmed], axis=0)
    else:
        result = pdf_vals

    return result

def spm_hrf(t, a1=6., d1=1., a2=16., d2=1., c=1./6, highres_dt=0.1):
    """Compute SPM canonical HRF(s)."""
    from .utils.backend import interp_regular_1d_grid

    t = ops.convert_to_tensor(t, dtype='float32')
    a1 = ops.convert_to_tensor(a1, dtype='float32')
    d1 = ops.convert_to_tensor(d1, dtype='float32')
    a2 = ops.convert_to_tensor(a2, dtype='float32')
    d2 = ops.convert_to_tensor(d2, dtype='float32')
    c = ops.convert_to_tensor(c, dtype='float32')

    t_min = ops.min(t)
    t_max = ops.max(t)
    n_steps = ops.cast(ops.ceil((t_max - t_min) / highres_dt) + 1, 'int32')
    t_hr = t_min + ops.arange(n_steps, dtype='float32') * highres_dt
    t_hr_2d = ops.expand_dims(t_hr, 1)

    hrf1_hr = gamma_pdf_with_loc(t_hr_2d, a1, d1, highres_dt, loc=highres_dt)
    hrf2_hr = gamma_pdf_with_loc(t_hr_2d, a2, d2, highres_dt, loc=highres_dt)
    hrf_hr = hrf1_hr - c * hrf2_hr
    hrf_hr /= ops.sum(hrf_hr, axis=0, keepdims=True)

    x_ref_max = t_min + highres_dt * ops.cast(n_steps - 1, 'float32')
    return interp_regular_1d_grid(t, t_min, x_ref_max, hrf_hr)

class SPMHRFModel(HRFModel):
    """Canonical SPM-style HRF parameterized by delay/dispersion bounds."""

    parameter_labels = ['hrf_delay', 'hrf_dispersion']
    n_parameters = 2

    def __init__(self, tr, unique_hrfs=False, highres_dt=0.1, time_length=32., onset=0.,
                 delay=6., undershoot=16., dispersion=1.,
                 u_dispersion=1., ratio=0.167,
                 min_hrf_delay=3., max_hrf_delay=7.0,
                 min_dispersion=0.3, max_dispersion=2.0):
        self.tr = tr
        self.highres_dt = highres_dt
        self.time_length = time_length
        self.onset = onset
        self.hrf_delay = delay
        self.undershoot = undershoot
        self.hrf_dispersion = dispersion
        self.u_dispersion = u_dispersion
        self.ratio = ratio
        self.min_hrf_delay = min_hrf_delay
        self.max_hrf_delay = max_hrf_delay
        self.min_dispersion = min_dispersion
        self.max_dispersion = max_dispersion
        self.dt = self.tr
        self.time_stamps = np.linspace(1e-4, self.time_length,
                                      np.rint(float(self.time_length) / self.dt).astype(np.int32)).astype(np.float32)
        self.time_stamps -= self.onset

        self.transformations = [
            bounded_sigmoid_transform(self.min_hrf_delay, self.max_hrf_delay),
            bounded_sigmoid_transform(self.min_dispersion, self.max_dispersion),
        ]

        super().__init__(unique_hrfs=unique_hrfs)

    def get_hrf(self, hrf_delay=None, hrf_dispersion=None):

        if hrf_delay is None:
            hrf_delay = self.hrf_delay

        if hrf_dispersion is None:
            hrf_dispersion = self.hrf_dispersion

        peak_shape = hrf_delay / hrf_dispersion + 1.
        undershoot_shape = self.undershoot / self.u_dispersion
        hrf = spm_hrf(self.time_stamps,
                      a1=peak_shape, d1=hrf_dispersion,
                      a2=undershoot_shape, d2=self.u_dispersion,
                      c=self.ratio,
                      highres_dt=self.highres_dt)
        return hrf


class SPMHRFDerivativeModel(HRFModel):
    parameter_labels = ['delay_weight', 'dispersion_weight']
    n_parameters = 2

    def __init__(self, tr, unique_hrfs=False, highres_dt=0.1, time_length=32., onset=0.,
                 delay=6., dispersion=1., undershoot=16., u_dispersion=1., ratio=0.167,
                 max_weight=2.0):
        self.tr = tr
        self.highres_dt = highres_dt
        self.time_length = time_length
        self.onset = onset
        self.delay = delay
        self.dispersion = dispersion
        self.undershoot = undershoot
        self.u_dispersion = u_dispersion
        self.ratio = ratio
        self.max_weight = max_weight
        self.time_stamps = np.linspace(1e-4, self.time_length,
                                       np.rint(float(self.time_length) / self.tr).astype(np.int32)).astype(np.float32)
        self.time_stamps -= self.onset
        self.time_stamps = self.time_stamps[:, np.newaxis]
        self.transformations = [
            bounded_sigmoid_transform(-self.max_weight, self.max_weight),
            bounded_sigmoid_transform(-self.max_weight, self.max_weight)
        ]
        super().__init__(unique_hrfs=unique_hrfs)

    def get_hrf(self, delay_weight=0., dispersion_weight=0.):
        base_hrf = spm_hrf(self.time_stamps,
                           a1=self.delay / self.dispersion + 1., d1=self.dispersion,
                           a2=self.undershoot / self.u_dispersion, d2=self.u_dispersion,
                           c=self.ratio,
                           highres_dt=self.highres_dt)

        eps = 1e-2

        d_delay = spm_hrf(self.time_stamps,
                          a1=(self.delay + eps) / self.dispersion + 1., d1=self.dispersion,
                          a2=self.undershoot / self.u_dispersion, d2=self.u_dispersion,
                          c=self.ratio,
                          highres_dt=self.highres_dt)
        derivative_delay = (d_delay - base_hrf) / eps

        d_disp = spm_hrf(self.time_stamps,
                         a1=self.delay / (self.dispersion + eps) + 1., d1=self.dispersion + eps,
                         a2=self.undershoot / self.u_dispersion, d2=self.u_dispersion,
                         c=self.ratio,
                         highres_dt=self.highres_dt)
        derivative_disp = (d_disp - base_hrf) / eps

        hrf = base_hrf \
              + delay_weight * derivative_delay \
              + dispersion_weight * derivative_disp

        return hrf


class CustomHRFModel(HRFModel):

    def __init__(self, hrf):
        self.hrf = hrf.astype(np.float32)
        self.oversampling = 1.
