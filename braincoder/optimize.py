import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm import tqdm
from .utils import format_data, format_parameters, format_paradigm
import logging


class ParameterOptimizer(object):

    def __init__(self, model, data, paradigm, log_dir=False, progressbar=True):
        self.model = model
        self.data = data
        self.paradigm = paradigm

        self.progressbar = True

        self.log_dir = log_dir

        if log_dir is None:
            log_dir = op.abspath('logs/fit')

        if log_dir is not False:
            if not op.exists(log_dir):
                os.makedirs(log_dir)
            self.summary_writer = tf.summary.create_file_writer(op.join(log_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def fit(self, max_n_iterations=1000,
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
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')
        elif hasattr(init_pars, 'values'):
            init_pars = init_pars.values.astype(np.float32)
        
        init_pars = self.model._transform_parameters_backward(init_pars)

        parameters = tf.Variable(initial_value=init_pars,
                                 shape=(n_voxels, n_pars), name='estimated_parameters', dtype=tf.float32)

        ssq_data = tf.math.reduce_variance(y, 0)
        ssq_data = tf.clip_by_value(ssq_data, 1e-6, 1e12)

        pbar = tqdm(range(max_n_iterations))

        if confounds is not None:
            # n_voxels x 1 x n_timepoints x n variables
            confounds = tf.repeat(
                confounds[tf.newaxis, tf.newaxis, :, :], n_voxels, 0)

        if store_intermediate_parameters:
            intermediate_parameters = []

        mean_r2s = []

        for step in pbar:
            try:
                with tf.GradientTape() as t:

                    untransformed_parameters = self.model._transform_parameters_forward(
                        parameters)

                    predictions_ = self.model._predict(self.paradigm,
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
                    ssq = tf.clip_by_value(
                        tf.math.reduce_variance(residuals, 0), 1e-6, 1e12)
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

                if hasattr(self, 'summary_write'):
                    with self.summary_writer.as_default():
                        tf.summary.scalar('mean R2', mean_r2, step=step)

                if store_intermediate_parameters:
                    p = untransformed_parameters.numpy().T
                    intermediate_parameters.append(
                        np.reshape(p, np.prod(p.shape)))
                    intermediate_parameters[-1] = np.concatenate(
                        (intermediate_parameters[-1], r2), 0)

                opt.apply_gradients(zip(gradients, trainable_variables))

                pbar.set_description(f'Mean R2: {mean_r2:0.5f}')


            except Exception as e:
                logging.log(f'Exception at step {step}: {e}')

        if store_intermediate_parameters:
            columns = pd.MultiIndex.from_product([self.model.parameter_labels + ['r2'],
                                                  np.arange(n_voxels)],
                                                 names=['parameter', 'voxel'])

            self.intermediate_parameters = pd.DataFrame(intermediate_parameters,
                                                        columns=columns,
                                                        index=pd.Index(np.arange(len(intermediate_parameters)),
                                                                       name='step'))

        self.estimated_parameters = format_parameters(
            untransformed_parameters.numpy(), self.model.parameter_labels)
        self.prediction = pd.DataFrame(
            predictions.numpy(), columns=self.data.columns, index=self.data.index)
        self.r2 = pd.DataFrame(
            r2.numpy()[np.newaxis, :], columns=self.data.columns)

    def fit_grid(self, *args, **kwargs):

        # Calculate a proper chunk size for cutting up the parameter grid
        n_timepoints, n_voxels = self.data.shape
        max_array_elements = int(4e9 / 6)  # 8 GB?
        chunk_size = max_array_elements / n_voxels / n_timepoints
        chunk_size = int(kwargs.pop('chunk_size', chunk_size))
        print(f'Working with chunk size of {chunk_size}')

        # Make sure that ranges for all parameters are given ing
        # *args or **kwargs
        if len(args) == len(self.model.parameter_labels):
            kwargs = dict(zip(self.model.parameter_labels, args))

        if not list(kwargs.keys()) == self.model.parameter_labels:
            raise ValueError(
                f'Please provide parameter ranges for all these parameters: {self.model.parameter_labels}')



        def _create_grid(model, *args):
            parameters = pd.MultiIndex.from_product(
                args, names=model.parameter_labels).to_frame(index=False)
            return parameters

        grid_args = [kwargs[key] for key in self.model.parameter_labels]
        par_grid = _create_grid(self.model, *grid_args).astype(np.float32)

        # Add chunks to the parameter columns, to process them chunk-wise and save memory
        par_grid = par_grid.set_index(
            pd.Index(par_grid.index // chunk_size, name='chunk'), append=True)

        logging.info('Built grid of {len(par_grid)} parameter settings...')

        @tf.function
        def _get_ssq_for_predictions(par_grid, data, model, paradigm):
            grid_predictions = model._predict(paradigm, par_grid, None)

            # time x voxels x parameters
            ssq = tf.math.reduce_variance(
                grid_predictions[:, tf.newaxis, :] - data[:, :, tf.newaxis], 0)
            # ssq = pd.DataFrame(ssq.numpy(), index=data.columns,
                               # columns=pd.MultiIndex.from_frame(par_grid))
            return ssq

        ssq = []
        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            ssq.append(_get_ssq_for_predictions(pg.values, self.data.values,
                self.model, self.paradigm.values))

        # ssq = pd.concat(ssq, 1)
        ssq = tf.concat(ssq, axis=1).numpy()
        ssq = pd.DataFrame(ssq, index=self.data.columns, columns=pd.MultiIndex.from_frame(par_grid))

        return ssq.idxmin(1).apply(lambda row: pd.Series(row, index=self.model.parameter_labels))

    def get_predictions(self, parameters):
        return self.model.predict(self.paradigm, parameters, None)

    def get_residuals(self, parameters):
        return self.data - self.get_predictions(parameters)

    def get_r2(self, parameters):
        residuals = self.get_residuals(parameters)
        return 1 - (residuals.var() / self.data.var())

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            self._data = format_data(data)

    @property
    def paradigm(self):
        return self._paradigm

    @paradigm.setter
    def paradigm(self, paradigm):
        self._paradigm = format_paradigm(paradigm)
