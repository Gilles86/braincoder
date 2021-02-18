import tensorflow as tf
import pandas as pd
import numpy as np


def norm(x, mu, sigma):
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel


def format_data(data):

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, tf.Tensor):
        data = data.numpy()

    return pd.DataFrame(data,
                        index=pd.Index(
                            np.arange(len(data)), name='time'),
                        columns=pd.Index(np.arange(data.shape[1]), name='voxel'))


def format_paradigm(paradigm):
    if paradigm is None:
        return None

    if paradigm.ndim == 1:
        paradigm = paradigm[:, np.newaxis]

    return pd.DataFrame(paradigm, index=pd.Index(range(len(paradigm)), name='time'),
                        columns=pd.Index(range(paradigm.shape[1]), name='stimulus dimension'))


def format_parameters(parameters, parameter_labels=None):

    if parameters is None:
        return None

    if parameter_labels is None:
        parameter_labels = [
            f'par{i+1}' for i in range(parameters.shape[1])]

    return pd.DataFrame(parameters,
                        columns=pd.Index(
                            parameter_labels, name='parameter'),
                        index=pd.Index(range(1, len(parameters) + 1), name='population'))
