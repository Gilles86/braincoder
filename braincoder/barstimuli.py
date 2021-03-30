from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
from .optimize import StimulusFitter
import logging
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from .utils.mcmc import cleanup_chain


class BarStimulusFitter(StimulusFitter):

    def __init__(self, data, model, grid_coordinates, omega, dof=None,
                 max_radius=None, max_width=None):

        self.data = data
        self.model = model
        self.grid_coordinates = pd.DataFrame(grid_coordinates)
        self.model.omega = omega
        self.model.dof = dof

        if max_radius is None:
            self.max_radius = np.sqrt(
                np.max(grid_coordinates['x'])**2 + np.max(grid_coordinates['y'])**2)
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

    def fit_grid(self, angle, radius, width, include_xy=True):

        data = self.data.values
        model = self.model
        parameters = self.model.parameters.values[np.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[np.newaxis, ...]
        grid_coordinates = self.grid_coordinates.values

        grid = pd.MultiIndex.from_product(
            [angle, radius, width], names=['angle', 'radius', 'width']).to_frame(index=False).astype(np.float32)

        logging.info('Built grid of {len(par_grid)} bar settings...')

        bars = make_bar_stimuli(grid_coordinates,
                                grid['angle'].values[np.newaxis, ...],
                                grid['radius'].values[np.newaxis, ...],
                                grid['width'].values[np.newaxis, ...])[0]

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
                                                   self.model.omega,
                                                   self.model.dof,
                                                   logp=True,
                                                   normalize=False)

            ll = tf.concat((tf.roll(ll, -hrf_shift, 1)
                            [:, :-hrf_shift], tf.ones((len(bars), hrf_shift)) * -1e12), 1)

        else:
            bars = bars[:, tf.newaxis, :]

            ll = self.model._likelihood(bars, data[tf.newaxis, ...], parameters, weights,
                                        self.model.omega, self.model.dof, logp=True)
        ll = pd.DataFrame(ll.numpy().T, columns=pd.MultiIndex.from_frame(grid))

        best_pars = ll.columns.to_frame().iloc[ll.values.argmax(1)]
        best_pars.index = self.data.index

        if include_xy:
            best_pars['x'] = np.cos(best_pars['angle']) * best_pars['radius']
            best_pars['y'] = np.sin(best_pars['angle']) * best_pars['radius']

        return best_pars.astype(np.float32)

    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            relevant_frames=None, rtol=1e-6, include_xy=True):

        if hasattr(init_pars, 'values'):
            init_pars = init_pars.values

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        if np.any(init_pars[:, 0] < -.5*np.pi):
            raise ValueError('All angles should be more than -1/2 pi radians')

        if np.any(init_pars[:, 0] > .5*np.pi):
            raise ValueError('All angles should be less than 1/2 pi radians')

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

        radius_bijector = tfb.Sigmoid(low=np.float32(-self.max_radius),
                                      high=np.float32(self.max_radius))

        width_bijector = tfb.Sigmoid(low=np.float32(0.0),
                                     high=np.float32(self.max_width))

        orient_x = tf.Variable(name='orient_x',
                               shape=len(init_pars),
                               initial_value=np.cos(init_pars[:, 0]))

        orient_y = tf.Variable(name='orient_y',
                               shape=len(init_pars),
                               initial_value=np.sin(init_pars[:, 0]))

        radius_ = tf.Variable(name='radius',
                              shape=len(init_pars),
                              initial_value=radius_bijector.inverse(init_pars[:, 1]))

        width_ = tf.Variable(name='width',
                             shape=len(init_pars),
                             initial_value=radius_bijector.inverse(init_pars[:, 2]))

        trainable_vars = [orient_x, orient_y, radius_, width_]

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        likelihood = self.build_likelihood_function(relevant_frames)

        for step in pbar:
            with tf.GradientTape() as tape:

                radius = radius_bijector.forward(radius_)
                width = width_bijector.forward(width_)
                angle = tf.math.atan(orient_y / orient_x)

                ll = likelihood(orient_x, orient_y, radius, width)[0]
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
        fitted_pars_ = np.concatenate([angle.numpy()[:, np.newaxis],
                                       radius.numpy()[:, np.newaxis],
                                       width.numpy()[:, np.newaxis]], axis=1)

        if relevant_frames is None:
            fitted_pars = pd.DataFrame(fitted_pars_, columns=['angle', 'radius', 'width'],
                                       index=self.data.index,
                                       dtype=np.float32)
        else:

            fitted_pars = pd.DataFrame(np.nan * np.zeros((self.data.shape[0], 3)), columns=['angle', 'radius', 'width'],
                                       index=self.data.index,
                                       dtype=np.float32)
            fitted_pars.iloc[relevant_frames, :] = fitted_pars_

        if include_xy:
            fitted_pars['x'] = np.cos(
                fitted_pars['angle']) * fitted_pars['radius']
            fitted_pars['y'] = np.sin(
                fitted_pars['angle']) * fitted_pars['radius']

        return fitted_pars

    def build_likelihood_function(self, relevant_frames=None, n_batches=1):

        data = self.data.values[tf.newaxis, ...]
        grid_coordinates = self.grid_coordinates
        model = self.model
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[tf.newaxis, ...]

        if relevant_frames is None:
            @tf.function
            def likelihood(orient_x, orient_y, radius, width):

                angle = tf.math.atan(orient_y/orient_x)

                bars = make_bar_stimuli(
                    grid_coordinates,
                    angle[tf.newaxis, ...],
                    radius[tf.newaxis, ...],
                    width[tf.newaxis, ...])

                ll = self.model._likelihood(
                    bars, data, parameters, weights, self.model.omega, dof=self.model.dof, logp=True)

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
            def likelihood(orient_x,
                           orient_y,
                           radius,
                           width):

                angle = tf.math.atan(orient_y/orient_x)
                bars = make_bar_stimuli(
                    grid_coordinates.values, angle, radius, width)

                stimulus = tf.scatter_nd(indices,
                                         bars,
                                         size_)

                ll = self.model._likelihood(
                    stimulus,  data, parameters, weights, self.model.omega, dof=self.model.dof, logp=True)

                sll = tf.reduce_sum(ll, 1)

                return sll

        return likelihood

    def sample_posterior(self,
                         init_pars,
                         n_chains,
                         relevant_frames=None,
                         step_size=0.0001,
                         n_burnin=10,
                         n_samples=10):

        init_pars = init_pars.astype(np.float32)
        init_pars['orient_x'] = np.cos(init_pars['angle'])
        init_pars['orient_y'] = np.sin(init_pars['angle'])

        @tf.function
        def sample_hmc(
                init_state,
                step_size,
                target_log_prob_fn,
                unconstraining_bijectors,
                target_accept_prob=0.75,
                num_steps=50,
                burnin=50):

            def trace_fn(_, pkr):
                return {
                    'log_prob': pkr.inner_results.inner_results.target_log_prob,
                    'diverging': pkr.inner_results.inner_results.has_divergence,
                    'is_accepted': pkr.inner_results.inner_results.is_accepted,
                    'accept_ratio': tf.exp(pkr.inner_results.inner_results.log_accept_ratio),
                    'leapfrogs_taken': pkr.inner_results.inner_results.leapfrogs_taken,
                    'step_size': pkr.inner_results.inner_results.step_size}

            hmc = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn,
                step_size=step_size)

            hmc = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=hmc,
                bijector=unconstraining_bijectors)

            adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=hmc,
                num_adaptation_steps=int(0.8 * burnin),
                target_accept_prob=target_accept_prob,
                # NUTS inside of a TTK requires custom getter/setter functions.
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    inner_results=pkr.inner_results._replace(
                        step_size=new_step_size)
                ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )

            # Sampling from the chain.
            return tfp.mcmc.sample_chain(
                num_results=burnin + num_steps,
                current_state=init_state,
                kernel=adaptive_sampler,
                trace_fn=trace_fn)

        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars.iloc[relevant_frames, :]

        unconstraining_bjs = [tfb.Identity(),
                              tfb.Identity(),
                              tfb.Sigmoid(low=-self.max_radius,
                                          high=self.max_radius),  # radius
                              tfb.Sigmoid(low=np.float32(0.0), high=self.max_width)]  # width

        state_mu = []
        state_std = []
        for ix, key in enumerate(['orient_x', 'orient_y', 'radius', 'width']):
            state_mu.append(tf.reduce_mean(
                unconstraining_bjs[ix].inverse(init_pars[key])))
            state_std.append(tf.math.reduce_std(
                unconstraining_bjs[ix].inverse(init_pars[key])))

        bijectors = [tfb.Chain([cb, tfb.Shift(sh), tfb.Scale(sc)])
                     for cb, sh, sc in zip(unconstraining_bjs, state_mu, state_std)]

        init_pars = init_pars[['orient_x', 'orient_y', 'radius', 'width']]

        initial_state = list(
            np.repeat(init_pars.values.T[:, np.newaxis, :], n_chains, 1))

        likelihood = self.build_likelihood_function(
            relevant_frames, n_batches=n_chains)

        step_size = [tf.fill([n_chains] + [1] * (len(s.shape) - 1),
                             tf.constant(0.001, np.float32)) for s in initial_state]
        samples, stats = sample_hmc(
            initial_state, step_size, likelihood, bijectors, num_steps=n_samples, burnin=n_burnin)

        angle = tf.math.atan(samples[1] / samples[0]).numpy()
        radius = samples[2].numpy()
        width = samples[3].numpy()

        if relevant_frames is None:
            frame_index = self.data.index
        else:
            frame_index = self.data.index[relevant_frames]

        angle = cleanup_chain(angle, 'angle', frame_index)
        width = cleanup_chain(width, 'width', frame_index)
        radius = cleanup_chain(radius, 'radius', frame_index)

        samples = pd.concat((angle, width, radius), 1)

        return samples, stats


@tf.function
def make_bar_stimuli(grid_coordinates, angle, radius, width, falloff_speed=1000.):

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
        (-distance + width[..., tf.newaxis] / 2) * falloff_speed)

    return bar
