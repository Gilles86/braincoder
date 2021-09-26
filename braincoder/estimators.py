import numpy as np
from .utils.math import log2
import tensorflow as tf


class MutualInformationEstimator(object):

    def __init__(self, model, stimulus_range, omega=None, dof=None):
        self.stimulus_range = np.array(stimulus_range).astype(np.float32)

        if self.stimulus_range.ndim == 1:
            self.stimulus_range = stimulus_range[:, np.newaxis].astype(
                np.float32)

        if self.stimulus_range.shape[1] > 1:
            raise NotImplementedError()

        self.model = model
        self.n_units = self.model.parameters.shape[0]

        self.p_stimulus = 1. / len(self.stimulus_range)

        self.omega = omega
        self.omega_chol = tf.linalg.cholesky(omega).numpy()
        self.dof = dof

        if self.model.weights is None:
            self.weights_ = None
        else:
            self.weights_ = self.model.weights.values[np.newaxis, ...]

    def estimate_mi(self, n=1000, uselog=True):

        resid_dist = self.model.get_residual_dist(
            self.n_units, self.omega_chol, self.dof)

        noise = resid_dist.sample(n)

        print('NOISE', noise)

        predictions = self.model._predict(self.stimulus_range[np.newaxis, ...],
                                          self.model.parameters.values[np.newaxis, ...],
                                          self.weights_)

        print('PREDICTIONS', predictions)
        # n samples x actual_stimuli x n_voxels
        neural_data = predictions + noise[:, tf.newaxis, :]

        print('NEURAL DATA', neural_data)
        p_stimulus = self.p_stimulus

        print('P_STIMULUS', p_stimulus)

        if uselog:

            # logp_joint = self.model._likelihood(self.stimulus_range[np.newaxis, :, :],
                                             # neural_data,
                                             # self.model.parameters.values[np.newaxis, ...],
                                             # self.weights_,
                                             # self.omega_chol,
                                             # self.dof,
                                             # logp=True)

            logp_joint = resid_dist.log_prob(noise)[:, tf.newaxis]

            # n x n_stimuli
            print('logp_joint', logp_joint)
            p_joint = np.exp(logp_joint, dtype=np.float128) * p_stimulus
            print('p_joint', p_joint)

            residuals = neural_data[:, :, tf.newaxis, :] - \
                predictions[:, tf.newaxis, :, :]

            # Still needs to be summed over 2 dimension
            # n x n_stimuli x n_hypothetical stimuli
            logp_data = resid_dist.log_prob(residuals)
            p_data = np.exp(logp_data, dtype=np.float128)
            print('logp_data', logp_data)
            print('p_data', p_data)
            p_data = np.sum(p_data, 2) / p_data.shape[2]

            print(p_joint / p_data)

            mi = 1. / n * np.sum(p_joint * np.log2(p_joint / (p_data * np.float128(p_stimulus))))

            print(mi)

            return np.array([mi])


        else:
            p_joint = self.model._likelihood(self.stimulus_range[np.newaxis, :, :],
                                             neural_data,
                                             self.model.parameters.values[np.newaxis, ...],
                                             self.weights_,
                                             self.omega_chol,
                                             self.dof) * p_stimulus
            print('p_joint', p_joint)

            # n samples x n_simulated x n_hypothetical x n_voxels
            residuals = neural_data[:, :, tf.newaxis, :] - \
                predictions[:, tf.newaxis, :, :]

            p_data = resid_dist.prob(residuals)
            p_data = tf.reduce_sum(p_data, 2) / p_data.shape[2]

            mi = 1. / n * tf.reduce_sum(p_joint *
                                        log2(p_joint / (p_data * p_stimulus)))

        return mi.numpy()
