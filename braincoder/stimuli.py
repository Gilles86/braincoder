from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
from .optimize import StimulusFitter
import logging
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from .utils.mcmc import cleanup_chain, sample_hmc, Periodic


class CustomStimulusFitter(StimulusFitter):

    def __init__(self, data, model, stimulus,
                 omega, dof=None,
                 baseline_image=None):

        self.data = data
        self.model = model
        self.stimulus = stimulus
        self.model.omega = omega
        self.model.omega_chol = tf.linalg.cholesky(omega).numpy()
        self.model.dof = dof

        if baseline_image is None:
            self.baseline_image = None
        else:
            self.baseline_image = baseline_image.reshape(
                len(self.grid_coordinates))

    # Parameters is dictionary
    def fit_grid(self, parameters):

        assert(set(parameters.keys()) == set(self.stimulus.parameter_labels)
               ), f"Provide a dictionary with the following keys: {self.stimulus.parameter_labels}"

        grid = pd.MultiIndex.from_product(
            parameters.values(), names=parameters.keys()).to_frame(index=False).astype(np.float32)

        grid = grid[self.stimulus.parameter_labels]

        logging.info(f'Built grid of {len(grid)} bar settings...')

        images = self.stimulus.generate_images(grid)[0]

        return self._fit_grid(grid, images)

    def _fit_grid(self, grid, images):

        data = self.data.values
        model = self.model
        parameters = self.model.parameters.values[np.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[np.newaxis, ...]

        if hasattr(self.model, 'hrf_model'):

            images = tf.concat((tf.zeros((images.shape[0],
                                          1,
                                          images.shape[1])),
                                images[:, tf.newaxis, :],
                                tf.zeros((images.shape[0],
                                          len(self.model.hrf_model.hrf)-1,
                                          images.shape[1]))),
                               1)

            hrf_shift = np.argmax(model.hrf_model.hrf)

            ts_prediction = model._predict(images, parameters, weights)

            baseline = ts_prediction[:, 0, :]

            ts_prediction_summed_over_hrf = tf.reduce_sum(
                ts_prediction - baseline[:, tf.newaxis, :], 1) / tf.reduce_sum(model.hrf_model.hrf) + baseline

            ll = self.model._likelihood_timeseries(data[tf.newaxis, ...],
                                                   ts_prediction_summed_over_hrf[:,
                                                                                 tf.newaxis, :],
                                                   self.model.omega_chol,
                                                   self.model.dof,
                                                   logp=True,
                                                   normalize=False)

            ll = tf.concat((tf.roll(ll, -hrf_shift, 1)
                            [:, :-hrf_shift], tf.ones((len(images), hrf_shift)) * -1e12), 1)

        else:
            images = images[:, tf.newaxis, :]

            ll = self.model._likelihood(images, data[tf.newaxis, ...], parameters, weights,
                                        self.model.omega_chol, self.model.dof, logp=True)

        ll = pd.DataFrame(ll.numpy().T, columns=pd.MultiIndex.from_frame(grid))

        best_pars = ll.columns.to_frame().iloc[ll.values.argmax(1)]
        best_pars.index = self.data.index

        return best_pars

    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            fit_vars=None,
            relevant_frames=None, rtol=1e-7, parameterization='xy', include_xy=True):

        model_vars = []
        trainable_vars = []

        if fit_vars is None:
            fit_vars = self.stimulus.parameter_labels

        if hasattr(init_pars, 'values'):
            init_pars = init_pars[self.stimulus.parameter_labels].values

        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars[relevant_frames, :]

        for label, bijector, values in zip(self.stimulus.parameter_labels, self.stimulus.bijectors, init_pars.T):
            assert(np.all(~np.isnan(bijector.inverse(values)))
                   ), f'Problem with init values of {label} (nans)'
            assert(np.all(np.isfinite(bijector.inverse(values)))
                   ), f'Problem with init values of {label} (infinte values)'

            tf_var = tf.Variable(name=label,
                                 shape=values.shape,
                                 initial_value=bijector.inverse(values))

            model_vars.append(tf_var)

            if label in fit_vars:
                trainable_vars.append(tf_var)

        likelihood = self.build_likelihood_function(relevant_frames)

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        for step in pbar:
            with tf.GradientTape() as tape:

                untransformed_pars = [bijector.forward(
                    par) for bijector, par in zip(self.stimulus.bijectors, model_vars)]
                ll = likelihood(*untransformed_pars)[0]
                cost = -ll

            gradients = tape.gradient(cost,
                                      trainable_vars)

            opt.apply_gradients(zip(gradients, trainable_vars))

            pbar.set_description(f'LL: {ll:6.4f}')

            self.costs[step] = cost.numpy()

            previous_cost = self.costs[np.max((step - lag, 0))]
            if step > min_n_iterations:
                if np.sign(previous_cost) == np.sign(cost):
                    if np.sign(cost) == 1:
                        if (cost / previous_cost) > 1 - rtol:
                            break
                    else:
                        if (cost / previous_cost) < 1 - rtol:
                            break

        fitted_pars_ = np.concatenate(
            [par.numpy()[:, np.newaxis] for par in untransformed_pars], axis=1)

        if relevant_frames is None:
            fitted_pars = pd.DataFrame(fitted_pars_, columns=self.stimulus.parameter_labels,
                                       index=self.data.index,
                                       dtype=np.float32)
        else:

            fitted_pars = pd.DataFrame(np.nan * np.zeros((self.data.shape[0], len(model_vars))), columns=self.stimulus.parameter_labels,
                                       index=self.data.index,
                                       dtype=np.float32)
            fitted_pars.iloc[relevant_frames, :] = fitted_pars_

        print(fitted_pars)

        return fitted_pars

    def build_likelihood_function(self, relevant_frames=None, falloff_speed=1000., n_batches=1):

        data = self.data.values[tf.newaxis, ...]
        grid_coordinates = self.stimulus.grid_coordinates.values
        model = self.model
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[tf.newaxis, ...]

        if self.baseline_image is None:
            @tf.function
            def add_baseline(images):
                return images
        else:
            print('Including base image (e.g., fixation image) into estimation')

            @tf.function
            def add_baseline(images):
                images = images + self.baseline_image[tf.newaxis, :]
                images = tf.clip_by_value(images, 0, self.max_intensity)
                return bars

        if relevant_frames is None:

            @tf.function
            def likelihood(*pars):
                images = self.stimulus._generate_images(*pars)
                images = add_baseline(images)
                ll = self.model._likelihood(
                    images, data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                return tf.reduce_sum(ll, 1)

        else:

            relevant_frames = tf.constant(relevant_frames, tf.int32)

            size_ = (n_batches, data.shape[1], len(grid_coordinates))
            size_ = tf.constant(size_, dtype=tf.int32)

            time_ix, batch_ix = np.meshgrid(relevant_frames, range(n_batches))
            indices = np.zeros(
                (n_batches, len(relevant_frames), 2), dtype=np.int32)
            indices[..., 0] = batch_ix
            indices[..., 1] = time_ix

            @tf.function
            def likelihood(*pars):

                images = self.stimulus._generate_images(*pars)

                images = tf.scatter_nd(indices,
                                       images,
                                       size_)

                images = add_baseline(images)

                ll = self.model._likelihood(
                    images,  data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                return tf.reduce_sum(ll, 1)

        return likelihood


class SzinteStimulus(object):

    parameter_labels = ['x', 'width', 'height']

    def __init__(self, grid_coordinates, max_width=None, max_height=None, intensity=1.0):

        self.grid_coordinates = pd.DataFrame(
            grid_coordinates, columns=['x', 'y'])
        self.intensity = intensity

        self.min_x, self.max_x = self.grid_coordinates['x'].min(
        ), self.grid_coordinates['x'].max()

        if max_width is None:
            max_width = self.max_x - self.min_x

        if max_height is None:
            max_height = self.grid_coordinates['y'].max(
            ) - self.grid_coordinates['y'].min()

        self.max_width = max_width
        self.max_height = max_height

        self.bijectors = [Periodic(low=self.min_x - self.max_width/2.,
                                   high=self.max_x + self.max_width/2.),  # x
                          tfb.Sigmoid(low=np.float32(0.0),
                                      high=self.max_width),  # width
                          tfb.Sigmoid(low=np.float32(0.0), high=self.max_height)]  # height

    def generate_images(self, parameters, falloff_speed=1000):
        return self._generate_images(parameters['x'].values,
                                     parameters['width'].values,
                                     parameters['height'].values,
                                     falloff_speed)

    def _generate_images(self, x, width, height, falloff_speed=1000):
        return make_aperture_stimuli(self.grid_coordinates.values,
                                     x, width, height, falloff_speed, self.intensity)


def make_aperture_stimuli(grid_coordinates, x, width, height, falloff_speed=1000., intensity=1.0):

    x_ = grid_coordinates[:, 0]
    y_ = grid_coordinates[:, 1]

    x_distance = tf.abs(x[..., tf.newaxis] - x_[tf.newaxis, tf.newaxis, ...])
    y_distance = tf.abs(y_[tf.newaxis, tf.newaxis, ...])

    bar_x = tf.math.sigmoid(
        (-x_distance + width[..., tf.newaxis] / 2) *
        falloff_speed)

    bar_y = tf.math.sigmoid(
        (-y_distance + height[..., tf.newaxis] / 2) *
        falloff_speed)

    return bar_x*bar_y*intensity
