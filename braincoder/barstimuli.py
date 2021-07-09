from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
from .optimize import StimulusFitter
import logging
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from .utils.mcmc import cleanup_chain, sample_hmc, Periodic


class BarStimulusFitter(StimulusFitter):

    def __init__(self, data, model, grid_coordinates, omega, dof=None,
                 max_radius=None, max_width=None, max_intensity=1.0):

        self.data = data
        self.model = model
        self.grid_coordinates = pd.DataFrame(
            grid_coordinates, columns=['x', 'y'])
        self.model.omega = omega
        self.model.omega_chol = tf.linalg.cholesky(omega).numpy()
        self.model.dof = dof
        self.max_intensity = max_intensity

        self.min_x, self.max_x = self.grid_coordinates['x'].min(), self.grid_coordinates['x'].max()
        self.min_y, self.max_y = self.grid_coordinates['y'].min(), self.grid_coordinates['y'].max()

        if max_radius is None:
            self.max_radius = np.sqrt(
                np.max(self.grid_coordinates['x'])**2 + np.max(self.grid_coordinates['y'])**2)
        else:
            self.max_radius = max_radius

        if max_width is None:
            self.max_width = .5 * self.max_radius
        else:
            self.max_width = max_width

        self.max_radius = np.float32(self.max_radius)
        self.max_width = np.float32(self.max_width)

        if self.model.weights is None:
            self.model.weights

    def fit_grid(self, x, y, width, include_xy=True):

        grid = pd.MultiIndex.from_product(
            [x, y, width], names=['x', 'y', 'width']).to_frame(index=False).astype(np.float32)

        logging.info('Built grid of {len(par_grid)} bar settings...')

        bars = make_bar_stimuli(self.grid_coordinates.values,
                                grid['x'].values[np.newaxis, ...],
                                grid['y'].values[np.newaxis, ...],
                                grid['width'].values[np.newaxis, ...],
                                intensity=self.max_intensity)[0]

        return self._fit_grid(grid, bars)

    def fit_grid2(self, angle, radius, width, include_xy=True):

        grid = pd.MultiIndex.from_product(
            [angle, radius, width], names=['angle', 'radius', 'width']).to_frame(index=False).astype(np.float32)

        logging.info('Built grid of {len(par_grid)} bar settings...')

        bars = make_bar_stimuli2(self.grid_coordinates.values,
                                grid['angle'].values[np.newaxis, ...],
                                grid['radius'].values[np.newaxis, ...],
                                grid['width'].values[np.newaxis, ...],
                                intensity=self.max_intensity)[0]
        return self._fit_grid(grid, bars)


    def _fit_grid(self, grid, bars):

        data = self.data.values
        model = self.model
        parameters = self.model.parameters.values[np.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[np.newaxis, ...]

        if hasattr(self.model, 'hrf_model'):

            bars = tf.concat((tf.zeros((bars.shape[0],
                                        1,
                                        bars.shape[1])),
                              bars[:, tf.newaxis, :],
                              tf.zeros((bars.shape[0],
                                        len(self.model.hrf_model.hrf)-1,
                                        bars.shape[1]))),
                             1)

            hrf_shift = np.argmax(model.hrf_model.hrf)

            ts_prediction = model._predict(bars, parameters, weights)

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
                            [:, :-hrf_shift], tf.ones((len(bars), hrf_shift)) * -1e12), 1)

        else:
            bars = bars[:, tf.newaxis, :]

            ll = self.model._likelihood(bars, data[tf.newaxis, ...], parameters, weights,
                                        self.model.omega_chol, self.model.dof, logp=True)
        ll = pd.DataFrame(ll.numpy().T, columns=pd.MultiIndex.from_frame(grid))

        best_pars = ll.columns.to_frame().iloc[ll.values.argmax(1)]
        best_pars.index = self.data.index

        if 'x' not in best_pars.columns:
            best_pars['x'] = np.cos(best_pars['angle']) * best_pars['radius']
            best_pars['y'] = np.sin(best_pars['angle']) * best_pars['radius']
        if 'angle' not in best_pars.columns:
            best_pars['angle'] = np.arctan2(best_pars['y'], best_pars['x'])
            best_pars['radius'] = np.sqrt(best_pars['y']**2 + best_pars['x']**2)

        return best_pars.astype(np.float32)


    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            relevant_frames=None, rtol=1e-6, include_xy=True):


        # init_pars: x, y, width

        if hasattr(init_pars, 'values'):
            init_pars = init_pars[['x', 'y', 'width']].values

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        if np.any(init_pars[:, 0] < self.min_x):
            raise ValueError(f'All x-values should not be less than {self.min_x}')

        if np.any(init_pars[:, 0] > self.max_x):
            raise ValueError(f'All x-values should not be more than {self.max_x}')

        if np.any(init_pars[:, 1] < self.min_y):
            raise ValueError(f'All y-values should not be less than {self.min_y}')

        if np.any(init_pars[:, 1] > self.max_y):
            raise ValueError(f'All y-values should not be more than {self.max_y}')

        if np.any(np.abs(init_pars[:, 2]) < 0.0):
            raise ValueError('All widths should be positive')

        if np.any(np.abs(init_pars[:, 2]) > self.max_width):
            raise ValueError(
                f'All widths should be less than {self.max_width}')

        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars[relevant_frames, :]


        x_bijector = Periodic(low=np.float32(self.min_x - self.max_width/2.),
                              high=np.float32(self.max_x + self.max_width/2.))

        y_bijector = Periodic(low=np.float32(self.min_y - self.max_width/2.),
                              high=np.float32(self.max_y + self.max_width/2.))

        width_bijector = tfb.Sigmoid(low=np.float32(0.0),
                                     high=np.float32(self.max_width))

        x_ = tf.Variable(name='x',
                        shape=len(init_pars),
                        initial_value=init_pars[:, 0])

        y_ = tf.Variable(name='y',
                        shape=len(init_pars),
                        initial_value=init_pars[:, 1])

        width_ = tf.Variable(name='width',
                             shape=len(init_pars),
                             initial_value=width_bijector.inverse(init_pars[:, 2]))

        trainable_vars = [x_, y_, width_]

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        likelihood = self.build_likelihood_function(relevant_frames)

        for step in pbar:
            with tf.GradientTape() as tape:

                x = x_bijector.forward(x_)
                y = y_bijector.forward(y_)
                width = width_bijector.forward(width_)

                ll = likelihood(x, y, width)[0]
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
        fitted_pars_ = np.concatenate([x.numpy()[:, np.newaxis],
                                       y.numpy()[:, np.newaxis],
                                       width.numpy()[:, np.newaxis]], axis=1)

        if relevant_frames is None:
            fitted_pars = pd.DataFrame(fitted_pars_, columns=['x', 'y', 'width'],
                                       index=self.data.index,
                                       dtype=np.float32)
        else:

            fitted_pars = pd.DataFrame(np.nan * np.zeros((self.data.shape[0], 3)), columns=['x', 'y', 'width'],
                                       index=self.data.index,
                                       dtype=np.float32)
            fitted_pars.iloc[relevant_frames, :] = fitted_pars_

        fitted_pars['angle'] = np.arctan2(fitted_pars['y'], fitted_pars['x'])
        fitted_pars['radius'] = np.sqrt(fitted_pars['y']**2 + fitted_pars['x']**2)

        return fitted_pars

    def build_likelihood_function(self, relevant_frames=None, falloff_speed=1000., n_batches=1):

        data = self.data.values[tf.newaxis, ...]
        grid_coordinates = self.grid_coordinates
        model = self.model
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[tf.newaxis, ...]

        if relevant_frames is None:
            @tf.function
            def likelihood(x, y, width):

                bars = make_bar_stimuli(
                    grid_coordinates,
                    x[tf.newaxis, ...],
                    y[tf.newaxis, ...],
                    width[tf.newaxis, ...],
                    falloff_speed=falloff_speed,
                    intensity=self.max_intensity)

                ll = self.model._likelihood(
                    bars, data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                return tf.reduce_sum(ll, 1)

        else:
            relevant_frames = tf.constant(relevant_frames, tf.int32)

            size_ = (n_batches, data.shape[1], len(grid_coordinates))
            size_ = tf.constant(size_, dtype=tf.int32)

            time_ix, batch_ix = np.meshgrid(relevant_frames, range(n_batches))
            indices = np.zeros((n_batches, len(relevant_frames), 2), dtype=np.int32)
            indices[..., 0] = batch_ix
            indices[..., 1] = time_ix

            @tf.function
            def likelihood(x,
                           y,
                           width):

                bars = make_bar_stimuli2(
                    grid_coordinates.values, x, y, width,
                    falloff_speed=falloff_speed,
                    intensity=self.max_intensity)

                stimulus = tf.scatter_nd(indices,
                                         bars,
                                         size_)

                ll = self.model._likelihood(
                    stimulus,  data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                sll = tf.reduce_sum(ll, 1)

                return sll

        return likelihood

    def sample_posterior(self,
                         init_pars,
                         n_chains,
                         relevant_frames=None,
                         step_size=0.0001,
                         n_burnin=10,
                         n_samples=10,
                         max_tree_depth=10,
                         unrolled_leapfrog_steps=1,
                         falloff_speed=1000.,
                         target_accept_prob=0.85):

        init_pars = init_pars.astype(np.float32)

        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars.iloc[relevant_frames, :]

        bijectors = [Periodic(low=self.min_x - self.max_width/2.,
                                  high=self.max_x + self.max_width/2.),  # x
                              Periodic(low=self.min_y - self.max_width/2.,
                                  high=self.max_y + self.max_width/2.),
                              tfb.Sigmoid(low=np.float32(0.0), high=self.max_width)]  # width

        init_pars = init_pars[['x', 'y', 'width']]

        initial_state = list(
            np.repeat(init_pars.values.T[:, np.newaxis, :], n_chains, 1))

        likelihood = self.build_likelihood_function(
            relevant_frames, falloff_speed=falloff_speed, n_batches=n_chains)

        step_size = [tf.fill([n_chains] + [1] * (len(s.shape) - 1),
                             tf.constant(step_size, np.float32)) for s in initial_state]
        samples, stats = sample_hmc(
            initial_state, step_size, likelihood, bijectors, num_steps=n_samples, burnin=n_burnin,
            target_accept_prob=target_accept_prob, unrolled_leapfrog_steps=unrolled_leapfrog_steps,
            max_tree_depth=max_tree_depth)

        print(samples)

        x = samples[0].numpy()
        y = samples[1].numpy()
        width = samples[2].numpy()

        if relevant_frames is None:
            frame_index = self.data.index
        else:
            frame_index = self.data.index[relevant_frames]

        x = cleanup_chain(x, 'x', frame_index)
        y = cleanup_chain(y, 'y', frame_index)
        width = cleanup_chain(width, 'width', frame_index)

        samples = pd.concat((x, y, width), 1)

        return samples, stats

@tf.function
def make_bar_stimuli(grid_coordinates, x, y, width, falloff_speed=1000., intensity=1.0):

    x_ = grid_coordinates[:, 0]
    y_ = grid_coordinates[:, 1]

    radius2 = x**2 + y**2
    radius = tf.math.sqrt(radius2)
    a = x / radius
    b = y / radius
    c = -radius2 / radius

    distance = tf.abs(a[..., tf.newaxis] * x_[tf.newaxis, tf.newaxis, ...] +
                      b[..., tf.newaxis] * y_[tf.newaxis, tf.newaxis, ...] +
                      c[..., tf.newaxis]) / tf.sqrt(a[..., tf.newaxis]**2 + b[..., tf.newaxis]**2)

    bar = tf.math.sigmoid(
        (-distance + width[..., tf.newaxis] / 2) * falloff_speed) * intensity

    return bar

@tf.function
def make_bar_stimuli2(grid_coordinates, angle, radius, width, falloff_speed=1000., intensity=1.0):

    # batch x stimulus x stimulus_dimension

    x = grid_coordinates[:, 0]
    y = grid_coordinates[:, 1]

    a = tf.cos(angle)
    b = tf.sin(angle)
    c = tf.sqrt(a**2 + b**2) * - radius

    distance = tf.abs(a[..., tf.newaxis] * x[tf.newaxis, tf.newaxis, ...] +
                      b[..., tf.newaxis] * y[tf.newaxis, tf.newaxis, ...] +
                      c[..., tf.newaxis]) / tf.sqrt(a[..., tf.newaxis]**2 + b[..., tf.newaxis]**2)

    bar = tf.math.sigmoid(
        (-distance + width[..., tf.newaxis] / 2) * falloff_speed) * intensity

    return bar

