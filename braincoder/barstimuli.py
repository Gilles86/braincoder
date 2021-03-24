from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
from .optimize import StimulusFitter
import logging


class BarStimulusFitter(StimulusFitter):

    def __init__(self, data, model, grid_coordinates, omega, dof=None):

        self.data = data
        self.model = model
        self.grid_coordinates = pd.DataFrame(grid_coordinates)
        self.model.omega = omega
        self.model.dof = dof

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
            best_pars['x'] = np.sin(best_pars['angle']) * best_pars['radius']
            best_pars['y'] = np.cos(best_pars['angle']) * best_pars['radius']

        return best_pars.astype(np.float32)

    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            rtol=1e-6, include_xy=True):

        data = self.data.values[tf.newaxis, ...]
        grid_coordinates = self.grid_coordinates
        model = self.model
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[tf.newaxis, ...]

        if hasattr(init_pars, 'values'):
            init_pars = init_pars.values

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        angle = tf.Variable(name='angle', shape=(
            data.shape[1],), initial_value=init_pars[:, 0])
        radius = tf.Variable(name='radius', shape=(
            data.shape[1],), initial_value=init_pars[:, 1])
        width = tf.Variable(name='width', shape=(
            data.shape[1],), initial_value=init_pars[:, 2])
        trainable_vars = [angle, radius, width]

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        for step in pbar:
            with tf.GradientTape() as tape:
                bars = make_bar_stimuli(
                    grid_coordinates,
                    angle[tf.newaxis, ...],
                    radius[tf.newaxis, ...],
                    width[tf.newaxis, ...])

                ll = self.model._likelihood(
                    bars, data, parameters, weights, self.model.omega, dof=self.model.dof, logp=True)

                sll = tf.reduce_sum(ll)
                cost = -sll

            gradients = tape.gradient(cost,
                                      trainable_vars)

            opt.apply_gradients(zip(gradients, trainable_vars))

            pbar.set_description(f'LL: {sll:6.4f}')

            self.costs[step] = cost.numpy()

            previous_cost = self.costs[np.max((step-lag, 0))]
            if step > min_n_iterations:
                if np.sign(previous_cost) == np.sign(cost):
                    if np.sign(cost) == 1:
                        if (cost / previous_cost) > 1 - rtol:
                            break
                    else:
                        if (cost / previous_cost) < 1 - rtol:
                            break
        fitted_pars = np.concatenate([angle.numpy()[:, np.newaxis],
                                      radius.numpy()[:, np.newaxis],
                                      width.numpy()[:, np.newaxis]], axis=1)

        fitted_pars = pd.DataFrame(fitted_pars, columns=['angle', 'radius', 'width'],
                                   index=self.data.index,
                                   dtype=np.float32)

        if include_xy:
            fitted_pars['x'] = np.sin(
                fitted_pars['angle']) * fitted_pars['radius']
            fitted_pars['y'] = np.cos(
                fitted_pars['angle']) * fitted_pars['radius']

        return fitted_pars


@tf.function
def make_bar_stimuli(grid_coordinates, angle, radius, width, falloff_speed=50.):

    # batch x stimulus x stimulus_dimension

    x = grid_coordinates[:, 0]
    y = grid_coordinates[:, 1]

    a = tf.sin(angle)
    b = tf.cos(angle)
    c = tf.sqrt(a**2 + b**2) * - radius

    distance = tf.abs(a[..., tf.newaxis] * x[tf.newaxis, tf.newaxis, ...] +
                      b[..., tf.newaxis] * y[tf.newaxis, tf.newaxis, ...] +
                      c[..., tf.newaxis]) / tf.sqrt(a[..., tf.newaxis]**2 + b[..., tf.newaxis]**2)

    bar = tf.math.sigmoid(
        (-distance + width[..., tf.newaxis] / 2) * falloff_speed)

    return bar
