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
                        columns=pd.Index(np.arange(data.shape[1]), name='voxel')).astype(np.float32)


def format_paradigm(paradigm):
    if paradigm is None:
        return None
    
    if isinstance(paradigm, pd.DataFrame):
        return paradigm

    if paradigm.ndim == 1:
        paradigm = paradigm[:, np.newaxis]

    return pd.DataFrame(paradigm, index=pd.Index(range(len(paradigm)), name='time'),
                        columns=pd.Index(range(paradigm.shape[1]), name='stimulus dimension')).astype(np.float32)


def format_parameters(parameters, parameter_labels=None):

    if isinstance(parameters, pd.DataFrame):
        return parameters.astype(np.float32)

    if parameters is None:
        return None

    if parameter_labels is None:
        parameter_labels = [
            f'par{i+1}' for i in range(parameters.shape[1])]

    return pd.DataFrame(parameters,
                        columns=pd.Index(
                            parameter_labels, name='parameter'),
                        index=pd.Index(range(1, len(parameters) + 1), name='population')).astype(np.float32)


def format_weights(weights):
    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            return weights
        else:
            return pd.DataFrame(weights,
                    index=pd.Index(range(1, len(weights) + 1), name='population'),
                    columns=pd.Index(np.arange(weights.shape[1]), name='voxel')).astype(np.float32)


def get_map(p):
    stimuli = p.columns.to_frame(index=False).T
    return stimuli.groupby(level=0).apply(lambda d: (p*d.values).sum(1) / p.sum(1)).T
