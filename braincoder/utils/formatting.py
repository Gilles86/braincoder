import pandas as pd
import numpy as np
import tensorflow as tf


def format_paradigm(paradigm):
    if paradigm is None:
        return None

    if isinstance(paradigm, pd.DataFrame):
        return paradigm

    if isinstance(paradigm, pd.Series):
        return paradigm.to_frame()

    if paradigm.ndim == 1:
        paradigm = paradigm[:, np.newaxis]
    elif paradigm.ndim > 2:
        paradigm = paradigm.reshape((paradigm.shape[0], -1))

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

    if type(parameter_labels) is list:
        parameter_labels = pd.Index(parameter_labels, name='parameter')

    return pd.DataFrame(parameters,
                        columns=parameter_labels,
                        index=pd.Index(range(len(parameters)), name='source')).astype(np.float32)


def format_weights(weights):
    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            return weights
        else:
            return pd.DataFrame(weights,
                                index=pd.Index(
                                    range(1, len(weights) + 1), name='population'),
                                columns=pd.Index(np.arange(weights.shape[1]), name='unit')).astype(np.float32)


def format_data(data):

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, tf.Tensor):
        data = data.numpy()

    return pd.DataFrame(data,
                        index=pd.Index(
                            np.arange(len(data)), name='time'),
                        columns=pd.Index(range(data.shape[1]), name='unit')).astype(np.float32)
