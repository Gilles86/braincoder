import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.math import sigmoid
from .utils import logit

class HRFModel(object):

    def __init__(self, unique_hrfs=False):
        self.set_unique_hrfs(unique_hrfs)

    def set_unique_hrfs(self, unique_hrfs):
        self.unique_hrfs = unique_hrfs

        if self.unique_hrfs:
            self._convolve = self._convolve_unique
        else:
            self._convolve = self._convolve_shared

    @tf.function
    def convolve(self, timeseries, **kwargs):

        hrf = self.get_hrf(**kwargs)

        if self.oversampling == 1:
            return self._convolve(timeseries, hrf=hrf)

        else:
            upsampled_timeseries = self._upsample(timeseries)
            cts = self._convolve(upsampled_timeseries, hrf=hrf)
            return self._downsample(cts)

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

    def _upsample(self, timeseries):
        new_length = len(timeseries) * self.oversampling
        timeseries_upsampled = tf.image.resize(timeseries[tf.newaxis, :, :, tf.newaxis],
                                               [new_length, timeseries.shape[1]])

        return tf.squeeze(timeseries_upsampled)

    def _downsample(self, upsampled_timeseries):
        new_length = len(upsampled_timeseries) // self.oversampling
        timeseries_downsampled = tf.image.resize(upsampled_timeseries[tf.newaxis, :, :, tf.newaxis],
                                                 [new_length, upsampled_timeseries.shape[1]])

        return tf.squeeze(timeseries_downsampled)

def gamma_pdf(t, a, d):
    """Compute the gamma probability density function at t."""
    return tf.pow(t, a - 1) * tf.exp(-t / d) / (tf.pow(d, a) * tf.exp(tf.math.lgamma(a)))

def gamma_pdf_with_loc(t, a, d, loc=0):
    # Applies location shift before evaluation
    t_shifted = t - loc
    return tf.where(
        t_shifted > 0,
        tf.pow(t_shifted, a - 1) * tf.exp(-t_shifted / d) / (tf.pow(d, a) * tf.exp(tf.math.lgamma(a))),
        tf.zeros_like(t)
    )

def spm_hrf(t, a1=6., d1=1., a2=16., d2=1., c=1./6, loc= 0.):
    """Compute the SPM canonical HRF at time points t."""
    hrf = gamma_pdf_with_loc(t, a1, d1, loc=loc) - c * gamma_pdf_with_loc(t, a2, d2, loc=loc)
    return hrf / tf.reduce_sum(hrf)

class SPMHRFModel(HRFModel):

    parameter_labels = ['hrf_delay', 'hrf_dispersion']
    n_parameters = 2

    def __init__(self, tr, unique_hrfs=False, oversampling=1, time_length=32., onset=0.,
                 delay=6., undershoot=16., dispersion=1.,
                 u_dispersion=1., ratio=0.167):

        self.tr = tr
        self.oversampling = oversampling
        self.time_length = time_length
        self.onset = onset
        self.hrf_delay = delay
        self.undershoot = undershoot
        self.hrf_dispersion = dispersion
        self.u_dispersion = u_dispersion
        self.ratio = ratio

        self.dt = self.tr / self.oversampling
        self.time_stamps = np.linspace(0, self.time_length,
                                  np.rint(float(self.time_length) / self.dt).astype(np.int32)).astype(np.float32)
        self.time_stamps -= self.onset

        # time x n_hrfs
        self.time_stamps = self.time_stamps[:, np.newaxis]

        super().__init__(unique_hrfs=unique_hrfs)

    def get_hrf(self, hrf_delay=6., hrf_dispersion=1.):
        peak_shape = hrf_delay / hrf_dispersion
        undershoot_shape = self.undershoot / self.u_dispersion
        hrf = spm_hrf(self.time_stamps, 
            a1=peak_shape, d1=hrf_dispersion,
            a2=undershoot_shape, d2=self.u_dispersion,
            c=self.ratio,
            loc=self.dt)
        return hrf

    @tf.function
    def _transform_parameters_forward(self, parameters):
        delay = sigmoid(parameters[:, 0][:, tf.newaxis])
        dispersion = sigmoid(parameters[:, 1][:, tf.newaxis])

        # Scale delay to be between [1.0, 10.0]
        delay = 9.0 * delay + 1.0

        # Scale dispersion to be between [0.75, 3.0]
        dispersion = 2.25 * dispersion + 0.75

        return tf.concat([delay, dispersion], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        delay = parameters[:, 0][:, tf.newaxis]
        delay = logit((delay - 1.0) / 9.0)

        dispersion = parameters[:, 1][:, tf.newaxis]
        dispersion = logit((dispersion - 0.75) / 2.25)

        return tf.concat([delay, dispersion], axis=1)

class CustomHRFModel(HRFModel):

    def __init__(self, hrf):
        self.hrf = hrf.astype(np.float32)
        self.oversampling = 1.
