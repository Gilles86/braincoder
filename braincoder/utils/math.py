import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np


def norm(x, mu, sigma):
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel

def norm2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=None):
    if rho is None:
        # Default to 0 covariance (independent x and y)
        rho = 0.0
    z = ((x - mu_x) ** 2 / sigma_x ** 2) + \
        ((y - mu_y) ** 2 / sigma_y ** 2) - \
        (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
    kernel = tf.math.exp(-z / (2 * (1 - rho ** 2)))
    return kernel

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return tf.clip_by_value(-tf.math.log(1. / x - 1.), -1e12, 1e12)

def logistic_transfer(x, lower_bound, upper_bound):
    return lower_bound + (upper_bound - lower_bound) / (1 + tf.exp(-x))

@tf.function
def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

@tf.function
def restrict_radians(x):
    x = x+np.pi
    return x - tf.floor(x / (2*np.pi)) * 2*np.pi - np.pi

def lognormalpdf_n(x, mu_n, sigma_n, normalize=False):

    denom = 1+sigma_n**2/mu_n**2

    part2 = tf.exp(-((tf.math.log(x)- tf.math.log(mu_n / tf.sqrt(denom)))**2 / (2*tf.math.log(denom))))

    if normalize:
        part1 = 1. / (x*tf.sqrt(2*np.pi*tf.math.log(denom)))
        return part1*part2
    else:
        return part2

def lognormal_pdf_mode_fwhm(x, mode, fwhm):

    sigma = 1./(tf.sqrt(2.*tf.math.log(2.))) * tf.math.asinh(fwhm/(mode*2.))
    sigma2 = sigma**2
    p = (mode / x) * tf.math.exp(.5*sigma2 - .5 *((tf.math.log(x/mode) - sigma2)**2)/sigma2)

    return p

def von_mises_pdf(x, mu, kappa):
    # Constants
    PI = tf.constant(np.pi)
    TWO_PI = tf.constant(2 * np.pi)

    # Calculate the PDF formula
    pdf = tf.exp(kappa * tf.cos(x - mu)) / (TWO_PI * tf.math.bessel_i0(kappa))

    return pdf

    import tensorflow as tf

# Aggressive softplus with alpha=100
alpha = 100
aggressive_softplus = lambda x: (1./alpha) * tf.math.softplus(alpha*x)
aggressive_softplus_inverse = lambda y: (1./alpha) * tfp.math.softplus_inverse(alpha * y)