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

        self.min_x, self.max_x = self.grid_coordinates['x'].min(
        ), self.grid_coordinates['x'].max()
        self.min_y, self.max_y = self.grid_coordinates['y'].min(
        ), self.grid_coordinates['y'].max()

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

    def fit_grid(self, x, y, width, parameterization='xy', include_xy=True):

        if parameterization == 'xy':

            grid = pd.MultiIndex.from_product(
                [x, y, width], names=['x', 'y', 'width']).to_frame(index=False).astype(np.float32)

            logging.info('Built grid of {len(par_grid)} bar settings...')

            bars = make_bar_stimuli(self.grid_coordinates.values,
                                    grid['x'].values[np.newaxis, ...],
                                    grid['y'].values[np.newaxis, ...],
                                    grid['width'].values[np.newaxis, ...],
                                    intensity=self.max_intensity)[0]

        else:
            raise NotImplementedError

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
            best_pars = get_angle_radius_from_xy(best_pars)

        return best_pars.astype(np.float32)

    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            relevant_frames=None, rtol=1e-7, parameterization='xy', include_xy=True):

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        # init_pars: x, y, width

        if parameterization == 'xy':
            if hasattr(init_pars, 'values'):
                init_pars = init_pars[['x', 'y', 'width']].values

            if np.any(init_pars[:, 0] < self.min_x):
                raise ValueError(
                    f'All x-values should not be less than {self.min_x}')

            if np.any(init_pars[:, 0] > self.max_x):
                raise ValueError(
                    f'All x-values should not be more than {self.max_x}')

            if np.any(init_pars[:, 1] < self.min_y):
                raise ValueError(
                    f'All y-values should not be less than {self.min_y}')

            if np.any(init_pars[:, 1] > self.max_y):
                raise ValueError(
                    f'All y-values should not be more than {self.max_y}')

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

            x_ = tf.Variable(name='x',
                             shape=len(init_pars),
                             initial_value=init_pars[:, 0])

            y_ = tf.Variable(name='y',
                             shape=len(init_pars),
                             initial_value=init_pars[:, 1])

        elif parameterization == 'angle':

            if hasattr(init_pars, 'values'):
                init_pars = init_pars[['angle', 'radius', 'width']].values

                if np.any(init_pars[:, 0] < -.5*np.pi):
                    raise ValueError(
                        'All angles should be more than -1/2 pi radians')

                if np.any(init_pars[:, 0] > .5*np.pi):
                    raise ValueError(
                        'All angles should be less than 1/2 pi radians')

                if np.any(np.abs(init_pars[:, 1]) > self.max_radius):
                    raise ValueError(
                        f'All radiuses should be within (-{self.max_radius}, {self.max_radius})')

                if np.any(np.abs(init_pars[:, 2]) < 0.0):
                    raise ValueError('All widths should be positive')

                if np.any(np.abs(init_pars[:, 2]) > self.max_width):
                    raise ValueError(
                        f'All widths should be less than {self.max_width}')

                if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
                    init_pars = init_pars[relevant_frames, :]

                init_pars[:, 0] = tf.clip_by_value(
                    init_pars[:, 0], -.5*np.pi + 1e-6, .5 * np.pi-1e-6)
                init_pars[:, 1] = tf.clip_by_value(
                    init_pars[:, 1], -self.max_radius + 1e-6, self.max_radius - 1e-6)
                init_pars[:, 2] = tf.clip_by_value(
                    init_pars[:, 2], 1e-6, self.max_width - 1e-6)

                orient_x = tf.Variable(name='orient_x',
                                       shape=len(init_pars),
                                       initial_value=np.cos(init_pars[:, 0]))

                orient_y = tf.Variable(name='orient_y',
                                       shape=len(init_pars),
                                       initial_value=np.sin(init_pars[:, 0]))

                radius_bijector = Periodic(low=np.float32(-self.max_radius),
                                           high=np.float32(self.max_radius))

                radius_ = tf.Variable(name='radius',
                                      shape=len(init_pars),
                                      initial_value=radius_bijector.inverse(init_pars[:, 1]))

        else:
            raise NotImplementedError

        width_bijector = tfb.Sigmoid(low=np.float32(0.0),
                                     high=np.float32(self.max_width))

        width_ = tf.Variable(name='width',
                             shape=len(init_pars),
                             initial_value=width_bijector.inverse(init_pars[:, 2]))

        if parameterization == 'xy':
            trainable_vars = [x_, y_, width_]
            bijectors = [x_bijector, y_bijector, width_bijector]
            par_names = ['x', 'y', 'width']
        elif parameterization == 'angle':
            trainable_vars = [orient_x, orient_y, radius_, width_]
            bijectors = [tfb.Identity(), tfb.Identity(),
                         radius_bijector, width_bijector]
            par_names = ['orient_x', 'orient_y', 'radius', 'width']

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        likelihood = self.build_likelihood_function(relevant_frames, parameterization=parameterization)

        for step in pbar:
            with tf.GradientTape() as tape:

                untransformed_pars = [bijector.forward(
                    par) for bijector, par in zip(bijectors, trainable_vars)]
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
            fitted_pars = pd.DataFrame(fitted_pars_, columns=par_names,
                                       index=self.data.index,
                                       dtype=np.float32)
        else:

            fitted_pars = pd.DataFrame(np.nan * np.zeros((self.data.shape[0], len(trainable_vars))), columns=par_names,
                                       index=self.data.index,
                                       dtype=np.float32)
            fitted_pars.iloc[relevant_frames, :] = fitted_pars_

        print(fitted_pars)
        if parameterization == 'xy':
            fitted_pars = get_angle_radius_from_xy(fitted_pars)

        elif parameterization == 'angle':
            print(relevant_frames)
            fitted_pars.at[relevant_frames, 'angle'] = tf.math.atan(orient_y / orient_x).numpy()
            fitted_pars['x'] = np.cos(
                fitted_pars['angle']) * fitted_pars['radius']
            fitted_pars['y'] = np.sin(
                fitted_pars['angle']) * fitted_pars['radius']


        return fitted_pars

    def build_likelihood_function(self, relevant_frames=None, falloff_speed=1000., parameterization='xy', n_batches=1):

        data = self.data.values[tf.newaxis, ...]
        grid_coordinates = self.grid_coordinates
        model = self.model
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[tf.newaxis, ...]

        if relevant_frames is None:

            if parameterization == 'xy':
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

            elif parameterization == 'angle':
                @tf.function
                def likelihood(orient_x, orient_y, radius, width):

                    angle = tf.math.atan(orient_y/orient_x)

                    bars = make_bar_stimuli2(
                        grid_coordinates,
                        angle[tf.newaxis, ...],
                        radius[tf.newaxis, ...],
                        width[tf.newaxis, ...],
                        falloff_speed=falloff_speed,
                        intensity=self.max_intensity)

                    ll = self.model._likelihood(
                        bars, data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                    return tf.reduce_sum(ll, 1)
            else:
                raise NotImplementedError

        else:
            relevant_frames = tf.constant(relevant_frames, tf.int32)

            size_ = (n_batches, data.shape[1], len(grid_coordinates))
            size_ = tf.constant(size_, dtype=tf.int32)

            time_ix, batch_ix = np.meshgrid(relevant_frames, range(n_batches))
            indices = np.zeros(
                (n_batches, len(relevant_frames), 2), dtype=np.int32)
            indices[..., 0] = batch_ix
            indices[..., 1] = time_ix

            if parameterization == 'xy':
                @tf.function
                def likelihood(x,
                               y,
                               width):

                    bars = make_bar_stimuli(
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

            elif parameterization == 'angle':

                @tf.function
                def likelihood(orient_x,
                               orient_y,
                               radius,
                               width):

                    angle = tf.math.atan(orient_y/orient_x)

                    bars = make_bar_stimuli2(
                        grid_coordinates,
                        angle[tf.newaxis, ...],
                        radius[tf.newaxis, ...],
                        width[tf.newaxis, ...],
                        falloff_speed=falloff_speed,
                        intensity=self.max_intensity)

                    stimulus = tf.scatter_nd(indices,
                                             bars,
                                             size_)

                    ll = self.model._likelihood(
                        stimulus, data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

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
                         target_accept_prob=0.85,
                         parameterization='xy'):

        init_pars = init_pars.astype(np.float32)

        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars.iloc[relevant_frames, :]

        
        if parameterization == 'xy':

            init_pars = init_pars[['x', 'y', 'width']]

            bijectors = [Periodic(low=self.min_x - self.max_width/2.,
                                  high=self.max_x + self.max_width/2.),  # x
                         Periodic(low=self.min_y - self.max_width/2.,
                                  high=self.max_y + self.max_width/2.),
                         tfb.Sigmoid(low=np.float32(0.0), high=self.max_width)]  # width

        initial_state = list(
            np.repeat(init_pars.values.T[:, np.newaxis, :], n_chains, 1))

        likelihood = self.build_likelihood_function(
            relevant_frames, falloff_speed=falloff_speed, n_batches=n_chains, parameterization=parameterization)

        step_size = [tf.fill([n_chains] + [1] * (len(s.shape) - 1),
                             tf.constant(step_size, np.float32)) for s in initial_state]
        samples, stats = sample_hmc(
            initial_state, step_size, likelihood, bijectors, num_steps=n_samples, burnin=n_burnin,
            target_accept_prob=target_accept_prob, unrolled_leapfrog_steps=unrolled_leapfrog_steps,
            max_tree_depth=max_tree_depth)

        if relevant_frames is None:
            frame_index = self.data.index
        else:
            frame_index = self.data.index[relevant_frames]

        cleaned_up_chains = [cleanup_chain(chain.numpy(), init_pars.columns[ix], frame_index) for ix, chain in enumerate(samples)]

        samples = pd.concat(cleaned_up_chains, 1)

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


def get_angle_radius_from_xy(d):
    d['angle'] = np.arctan2(d['y'], d['x'])
    d['radius'] = np.sqrt(d['y']**2 + d['x']**2)

    return constrain_angle(d)



def constrain_angle(d):

    pi5 = .5 * np.pi
    d['radius'].where(d['angle'].abs() < pi5, -d['radius'], inplace=True)
    d['angle'].where(d['angle'] < pi5, d['angle'] - np.pi, inplace=True)
    d['angle'].where(d['angle'] > -pi5, d['angle'] + np.pi, inplace=True)
    
    return d
