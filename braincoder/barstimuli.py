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

    def fit_grid(self, angle, radius, width):

        data = self.data.values
        model = self.model
        parameters = self.model.parameters.values
        weights = None if model.weights is None else model.weights.values
        grid_coordinates = self.grid_coordinates.values

        grid = pd.MultiIndex.from_product(
            [angle, radius, width], names=['angle', 'radius', 'width']).to_frame(index=False).astype(np.float32)

        logging.info('Built grid of {len(par_grid)} bar settings...')

        bars = self.make_bar_stimuli(grid_coordinates,
                                     grid['angle'], grid['radius'], grid['width'])

        ll = []
        for row in range(len(data)):
            ll.append(self.model._likelihood(bars[tf.newaxis, ...],
                                             data[[row]], parameters, weights, self.model.omega, dof=self.model.dof, logp=True).numpy())

        ll = pd.DataFrame(np.squeeze(
            ll), columns=pd.MultiIndex.from_frame(grid))

        best_pars = ll.columns.to_frame().iloc[ll.values.argmax(1)]
        best_pars.index = self.data.index

        return best_pars.astype(np.float32)

    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            rtol=1e-6):

        data = self.data.values
        grid_coordinates = self.grid_coordinates
        model = self.model
        parameters = self.model.parameters.values
        weights = None if model.weights is None else model.weights.values

        if hasattr(init_pars, 'values'):
            init_pars = init_pars.values

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        angle = tf.Variable(name='angle', shape=(
            len(data),), initial_value=init_pars[:, 0])
        radius = tf.Variable(name='radius', shape=(
            len(data),), initial_value=init_pars[:, 1])
        width = tf.Variable(name='width', shape=(
            len(data),), initial_value=init_pars[:, 2])
        trainable_vars = [angle, radius, width]

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        for step in pbar:
            with tf.GradientTape() as tape:
                bars = self.make_bar_stimuli(
                    grid_coordinates, angle, radius, width)

                ll = self.model._likelihood(
                    bars,  data, parameters, weights, self.model.omega, dof=self.model.dof, logp=True)

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

        return fitted_pars

    @tf.function
    def make_bar_stimuli(self, grid_coordinates, angle, radius, width, falloff_speed=50.):

        # stimuli: n_timepoints x n_stimuli x n_stimulus_features
        x = grid_coordinates[tf.newaxis, :, 0]
        y = grid_coordinates[tf.newaxis, :, 1]

        angle, radius, width = angle[:, tf.newaxis], radius[:,
                                                            tf.newaxis], width[:, tf.newaxis]

        a = tf.sin(angle)
        b = tf.cos(angle)
        c = tf.sqrt(a**2 + b**2) * -radius

        distance = tf.abs(a * x + b * y + c) / tf.sqrt(a**2 + b**2)

        bar = tf.math.sigmoid((-distance + width / 2) * falloff_speed)

        return bar
