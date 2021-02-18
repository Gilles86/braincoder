import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm import tqdm
from .utils import format_data, format_parameters


class ParameterOptimizer(object):

    def __init__(self, model, data, log_dir=None, progressbar=True):
        self.model = model
        self.data = data

        self.progressbar = True

        if log_dir is None:
            log_dir = op.abspath('logs/fit')
            if not op.exists(log_dir):
                os.makedirs(log_dir)

        self.summary_writer = tf.summary.create_file_writer(op.join(log_dir,
                                                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def optimize_parameters(self, paradigm, max_n_iterations=1000,
                            min_n_iterations=100,
                            init_pars=None,
                            confounds=None,
                            optimizer=None,
                            store_intermediate_parameters=True,
                            r2_atol=0.000001,
                            learning_rate=0.01):

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)


        y = self.data.values

        if optimizer is None:
            opt = tf.optimizers.Adam(learning_rate=learning_rate)

        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=paradigm, confounds=confounds)

        parameters = tf.Variable(initial_value=init_pars,
                                 shape=(n_voxels, n_pars), name='estimated_parameters', dtype=tf.float32)

        ssq_data = tf.math.reduce_variance(y, 0)

        pbar = tqdm(range(max_n_iterations))

        if confounds is not None:
            # n_voxels x 1 x n_timepoints x n variables
            confounds = tf.repeat(
                confounds[tf.newaxis, tf.newaxis, :, :], n_voxels, 0)

        if store_intermediate_parameters:
            intermediate_parameters = []

        mean_r2s = []

        for step in pbar:
            with tf.GradientTape() as t:

                untransformed_parameters = self.model._transform_parameters_forward(
                    parameters)

                predictions_ = self.model._predict(paradigm,
                                                   untransformed_parameters,
                                                   np.identity(n_voxels, dtype=np.float32))

                if confounds is None:
                    residuals = y - predictions_
                    predictions = predictions_
                else:
                    predictions = tf.transpose(predictions_)[
                        :, tf.newaxis, :, tf.newaxis]
                    X = tf.concat([predictions, confounds], -1)
                    beta = tf.linalg.lstsq(X, tf.transpose(
                        y)[:, tf.newaxis, :, tf.newaxis])
                    predictions = tf.transpose((X @ beta)[:, 0, :, 0])

                residuals = y - predictions
                ssq = tf.squeeze(tf.reduce_sum(residuals**2, ))
                ssq = tf.math.reduce_variance(residuals, 0)
                cost = tf.reduce_sum(ssq)

            trainable_variables = [parameters]
            gradients = t.gradient(cost, trainable_variables)
            r2 = (1 - (ssq / ssq_data))
            mean_r2 = tf.reduce_mean(r2)

            if step >= min_n_iterations:
                r2_diff = mean_r2 - mean_r2s[-1]
                if (r2_diff >= 0.0) & (r2_diff < r2_atol):
                    pbar.close()
                    break

            mean_r2s.append(mean_r2)

            with self.summary_writer.as_default():
                tf.summary.scalar('mean R2', mean_r2, step=step)

            if store_intermediate_parameters:
                p = untransformed_parameters.numpy().T
                intermediate_parameters.append(np.reshape(p, np.prod(p.shape)))
                intermediate_parameters[-1] = np.concatenate(
                    (intermediate_parameters[-1], r2), 0)

            opt.apply_gradients(zip(gradients, trainable_variables))

            pbar.set_description(f'Mean R2: {mean_r2:0.2f}')

            self.summary_writer.flush()

        if store_intermediate_parameters:
            columns = pd.MultiIndex.from_product([self.model.parameter_labels + ['r2'],
                                                  np.arange(n_voxels)],
                                                 names=['parameter', 'voxel'])

            self.intermediate_parameters = pd.DataFrame(intermediate_parameters,
                                                        columns=columns,
                                                        index=pd.Index(np.arange(len(intermediate_parameters)),
                                                                       name='step'))

        self.estimated_parameters = format_parameters(untransformed_parameters.numpy(), self.model.parameter_labels)
        self.prediction = pd.DataFrame(predictions.numpy(), columns=self.data.columns, index=self.data.index)
        self.r2 = pd.DataFrame(r2.numpy()[np.newaxis, :], columns=self.data.columns)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            self._data = format_data(data)
