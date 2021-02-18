import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class SPMHRFModel(object):

    def __init__(self, tr, oversampling=1, time_length=32., onset=0.,
                 delay=6, undershoot=16., dispersion=1.,
                 u_dispersion=1., ratio=0.167):

        self.tr = tr
        self.oversampling = oversampling
        self.time_length = time_length
        self.onset = onset
        self.delay = delay
        self.undershoot = undershoot
        self.dipsersion = dispersion
        self.u_dispersion = u_dispersion
        self.ratio = ratio

        dt = tr / oversampling
        time_stamps = np.linspace(0, time_length,
                                  np.rint(float(time_length) / dt).astype(np.int))
        time_stamps -= onset

        g1 = tfp.distributions.Gamma(concentration=delay / dispersion, rate=1.)
        shift = dt / dispersion
        over = g1.prob(time_stamps - shift)
        over = tf.where(tf.math.is_nan(over), tf.zeros_like(over), over)

        g2 = tfp.distributions.Gamma(
            concentration=undershoot / u_dispersion, rate=1.)
        shift_u = dt / u_dispersion
        under = g2.prob(time_stamps - shift_u)
        under = tf.where(tf.math.is_nan(under), tf.zeros_like(under), under)

        self.hrf = over - ratio * under

        self.hrf = self.hrf / tf.reduce_sum(self.hrf)

    @tf.function
    def convolve(self, timeseries):

        if self.oversampling == 1:
            return self._convolve(timeseries)

        else:
            upsampled_timeseries = self._upsample(timeseries)
            cts = self._convolve(upsampled_timeseries)
            return self._downsample(cts)

    def _convolve(self, timeseries):
        cts = tf.nn.conv1d(tf.transpose(timeseries)[..., tf.newaxis],
                           self.hrf[:, tf.newaxis, tf.newaxis],
                           stride=1,
                           padding='SAME')
        return tf.transpose(tf.squeeze(cts))

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
