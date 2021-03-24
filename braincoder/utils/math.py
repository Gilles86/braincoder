import tensorflow as tf
import pandas as pd
import numpy as np


def norm(x, mu, sigma):
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return tf.clip_by_value(-tf.math.log(1. / x - 1.), -1e12, 1e12)
