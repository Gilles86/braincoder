import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.math import sigmoid
from .utils import logit

class HRFModel(object):
    def set_unique_hrfs(self, unique_hrfs):
        self.unique_hrfs = unique_hrfs
        self._convolve = self._convolve_unique if unique_hrfs else self._convolve_shared

    def __init__(self, unique_hrfs=False):
        self.unique_hrfs = unique_hrfs
        self._convolve = self._convolve_unique if unique_hrfs else self._convolve_shared

    @tf.function
    def convolve(self, timeseries, **kwargs):
        hrf = self.get_hrf(**kwargs)
        return self._convolve(timeseries, hrf=hrf)

    def _convolve_shared(self, timeseries, hrf):
        # timeseries: returns: n_batch x n_timepoints x n_units
        # hrf: n_hrf_timepoints
        pad_ = tf.tile(timeseries[:, :1, :], [1, len(hrf), 1])
        timeseries_padded = tf.concat((pad_, timeseries), 1)

        cts = tf.nn.conv2d(timeseries_padded[:, :, :, tf.newaxis],
                           # THIS IS WEIRD TENSORFLOW BEHAVIOR
                           hrf[:, tf.newaxis,
                                    :, tf.newaxis][::-1],
                           strides=[1, 1],
                           padding='VALID')[:, :, :, 0]

        return cts[:, 1:, :]

    def _convolve_unique(self, timeseries, hrf):
        # timeseries: returns: n_batch x n_timepoints x n_units
        # hrf: n_hrf_timepoints x n_units
        print(timeseries.shape, hrf.shape)
        n, m = timeseries.shape[1], timeseries.shape[2]

        pad_ = tf.tile(timeseries[:, :1, :], [1, len(hrf), 1])
        timeseries_padded = tf.concat((pad_, timeseries), 1)

        # Reshape the timeseries data to 4D
        timeseries_4d = tf.reshape(timeseries_padded, [1, 1, n+len(hrf), m])  # Shape: [batch, in_height, in_width, in_channels]

        # Reshape the HRF filters to 4D
        hrf_filters_4d = tf.reshape(hrf, [1, len(hrf), m, 1])  # Shape: [filter_height, filter_width, in_channels, channel_multiplier]

        # Perform depthwise convolution
        convolved = tf.nn.depthwise_conv2d(
            input=timeseries_4d,
            filter=tf.reverse(hrf_filters_4d, axis=[1]),
            strides=[1, 1, 1, 1],
            padding='VALID'
        )

        convolved = convolved[:, :, 1:, :]

        return tf.reshape(convolved, [1, n, m])


import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow_probability as tfp

def gamma_pdf(t, a, d, eps=1e-6):
    """Numerically stable gamma PDF.

    Args:
        t: Tensor of times [T, M]
        a: Shape parameter (scalar or [1, M])
        d: Scale parameter (scalar or [1, M])
        eps: Small constant to avoid evaluating at t = 0

    Returns:
        Tensor of shape [T, M] with gamma pdf values
    """
    t = tf.maximum(t, eps)  # Ensure t > 0 for numerical stability
    coef = tf.pow(t, a - 1) * tf.exp(-t / d)
    denom = tf.pow(d, a) * tf.exp(tf.math.lgamma(a))
    return coef / denom


def gamma_pdf_with_loc(t, a, d, dt, loc=0.0):
    """
    Gamma PDF shifted by a delay 'loc', with zero-padding for pre-onset times.

    Args:
        t: Tensor of shape [T, M], time values
        a: Shape parameter (scalar or [1, M])
        d: Scale parameter (scalar or [1, M])
        dt: Time resolution (scalar float)
        loc: Delay offset (scalar float, in same units as t)

    Returns:
        Tensor of shape [T, M] representing the shifted and padded gamma PDF
    """
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    d = tf.convert_to_tensor(d, dtype=tf.float32)

    # Shift time by 'loc'
    shifted_t = t - loc

    # Only evaluate gamma PDF where t >= loc; avoid computing PDF on invalid (negative) inputs
    mask = shifted_t >= 0
    safe_t = tf.where(mask, shifted_t, tf.ones_like(shifted_t))  # dummy value for masked-out entries
    pdf_vals = tf.where(mask, gamma_pdf(safe_t, a, d), 0.0)

    # Convert 'loc' into number of time steps to pad
    loc_steps = int(round(float(loc) / float(dt)))

    if loc_steps > 0:
        # Shift the function forward in time by padding zeros at the beginning
        trimmed = pdf_vals[:-loc_steps, :]
        padding = tf.zeros([loc_steps, tf.shape(trimmed)[1]], dtype=pdf_vals.dtype)
        result = tf.concat([padding, trimmed], axis=0)
    else:
        # No shift needed
        result = pdf_vals

    # Optional check for debugging numerical issues
    # tf.debugging.check_numerics(result, "NaNs or Infs in result")

    return result

def spm_hrf(t, a1=6., d1=1., a2=16., d2=1., c=1./6, highres_dt=0.1):
    """
    Compute SPM canonical HRF(s).

    Args:
        t: tensor [T, M]
        a1, d1, a2, d2, c: scalars or tensors broadcastable to [T, M]
        dt: scalar time step

    Returns:
        Tensor [T, M] normalized HRFs
    """
    # All-TensorFlow version: compute at high-res, then interpolate to t using tfp.math.interp_regular_1d_grid
    t_min = tf.reduce_min(t)
    t_max = tf.reduce_max(t)
    n_steps = tf.cast(tf.math.ceil((t_max - t_min) / highres_dt) + 1, tf.int32)
    t_hr = t_min + tf.range(n_steps, dtype=tf.float32) * highres_dt  # [N]
    t_hr_2d = tf.expand_dims(t_hr, 1)  # [N, 1]

    # Compute high-res HRF
    hrf1_hr = gamma_pdf_with_loc(t_hr_2d, a1, d1, highres_dt, loc=highres_dt)
    hrf2_hr = gamma_pdf_with_loc(t_hr_2d, a2, d2, highres_dt, loc=highres_dt)
    hrf_hr = hrf1_hr - c * hrf2_hr  # [N, M]
    hrf_hr_sum = tf.reduce_sum(hrf_hr, axis=0, keepdims=True)
    hrf_hr_norm = hrf_hr / hrf_hr_sum  # [N, M]

    # t: [T], a1/d1/a2/d2/c: scalars or [M]
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    a1 = tf.convert_to_tensor(a1, dtype=tf.float32)
    d1 = tf.convert_to_tensor(d1, dtype=tf.float32)
    a2 = tf.convert_to_tensor(a2, dtype=tf.float32)
    d2 = tf.convert_to_tensor(d2, dtype=tf.float32)
    c = tf.convert_to_tensor(c, dtype=tf.float32)
    # High-res grid
    t_min = tf.reduce_min(t)
    t_max = tf.reduce_max(t)
    n_steps = tf.cast(tf.math.ceil((t_max - t_min) / highres_dt) + 1, tf.int32)
    t_hr = t_min + tf.range(n_steps, dtype=tf.float32) * highres_dt  # [N]
    t_hr_2d = tf.expand_dims(t_hr, 1)  # [N, 1]
    # Let TF broadcasting handle param shapes: if any param is [M], output will be [N, M]
    hrf1_hr = gamma_pdf_with_loc(t_hr_2d, a1, d1, highres_dt, loc=highres_dt)  # [N, M] or [N, 1]
    hrf2_hr = gamma_pdf_with_loc(t_hr_2d, a2, d2, highres_dt, loc=highres_dt)  # [N, M] or [N, 1]
    hrf_hr = hrf1_hr - c * hrf2_hr  # [N, M] or [N, 1]
    hrf_hr /= tf.reduce_sum(hrf_hr, axis=0, keepdims=True)
    # Interpolate: for each HRF, vectorized
    t_query = tf.expand_dims(t, 1)  # [T, 1]
    hrf_interp = tfp.math.interp_regular_1d_grid(
        x=t_query,
        x_ref_min=t_min,
        x_ref_max=t_min + highres_dt * tf.cast(n_steps - 1, tf.float32),
        y_ref=hrf_hr,
        axis=0,
        fill_value='constant_extension'
    )  # [T, M] or [T, 1]
    # Always return [T, M] (never [T, 1, M] or [T, 1])
    if hrf_interp.shape.rank == 1:
        hrf_interp = tf.expand_dims(hrf_interp, -1)
    if hrf_interp.shape.rank == 3:
        hrf_interp = tf.squeeze(hrf_interp, axis=1)
    return hrf_interp


def bounded_sigmoid_transform(min_val, max_val):
    def forward(x):
        return tf.sigmoid(x) * (max_val - min_val) + min_val
    def backward(y):
        y_scaled = (y - min_val) / (max_val - min_val)
        return tf.math.log(y_scaled / (1.0 - y_scaled))
    return (forward, backward)


class SPMHRFModel(HRFModel):


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
        self.dt = self.tr  # Use TR as time step unless oversampling is needed
        self.time_stamps = np.linspace(1e-4, self.time_length,
                                      np.rint(float(self.time_length) / self.dt).astype(np.int32)).astype(np.float32)
        self.time_stamps -= self.onset
        # 1D time vector
        self.transformations = [
            bounded_sigmoid_transform(self.min_hrf_delay, self.max_hrf_delay),
            bounded_sigmoid_transform(self.min_dispersion, self.max_dispersion)
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
        # Compute base HRF
        base_hrf = spm_hrf(self.time_stamps,
                           a1=self.delay / self.dispersion + 1., d1=self.dispersion,
                           a2=self.undershoot / self.u_dispersion, d2=self.u_dispersion,
                           c=self.ratio,
                           highres_dt=self.highres_dt)

        # Small perturbations for numerical derivatives
        eps = 1e-2

        # Temporal derivative: ∂HRF/∂delay
        d_delay = spm_hrf(self.time_stamps,
                          a1=(self.delay + eps) / self.dispersion + 1., d1=self.dispersion,
                          a2=self.undershoot / self.u_dispersion, d2=self.u_dispersion,
                          c=self.ratio,
                          highres_dt=self.highres_dt)
        derivative_delay = (d_delay - base_hrf) / eps

        # Dispersion derivative: ∂HRF/∂dispersion
        d_disp = spm_hrf(self.time_stamps,
                         a1=self.delay / (self.dispersion + eps) + 1., d1=self.dispersion + eps,
                         a2=self.undershoot / self.u_dispersion, d2=self.u_dispersion,
                         c=self.ratio,
                         highres_dt=self.highres_dt)
        derivative_disp = (d_disp - base_hrf) / eps

        # Linear combination
        hrf = base_hrf \
              + delay_weight * derivative_delay \
              + dispersion_weight * derivative_disp

        return hrf




class CustomHRFModel(HRFModel):

    def __init__(self, hrf):
        self.hrf = hrf.astype(np.float32)
        self.oversampling = 1.
