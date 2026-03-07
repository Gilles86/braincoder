import pandas as pd
import numpy as np
import keras
from keras import ops

from .backend import softplus_inverse


def norm(x, mu, sigma):
    kernel = ops.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel

def norm2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=None):
    if rho is None:
        rho = 0.0
    z = ((x - mu_x) ** 2 / sigma_x ** 2) + \
        ((y - mu_y) ** 2 / sigma_y ** 2) - \
        (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
    kernel = ops.exp(-z / (2 * (1 - rho ** 2)))
    return kernel

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return ops.clip(-ops.log(1. / x - 1.), -1e12, 1e12)

def logistic_transfer(x, lower_bound, upper_bound):
    return lower_bound + (upper_bound - lower_bound) / (1 + ops.exp(-x))

def log2(x):
    return ops.log(x) / ops.log(ops.convert_to_tensor(2.))

def restrict_radians(x):
    x = x + np.pi
    return x - ops.floor(x / (2*np.pi)) * 2*np.pi - np.pi

def lognormalpdf_n(x, mu_n, sigma_n, normalize=False):

    denom = 1 + sigma_n**2/mu_n**2

    part2 = ops.exp(-((ops.log(x) - ops.log(mu_n / ops.sqrt(denom)))**2 / (2*ops.log(denom))))

    if normalize:
        part1 = 1. / (x*ops.sqrt(2*np.pi*ops.log(denom)))
        return part1*part2
    else:
        return part2

def lognormal_pdf_mode_fwhm(x, mode, fwhm):

    sigma = 1./(ops.sqrt(2.*ops.log(ops.convert_to_tensor(2.)))) * ops.arcsinh(fwhm/(mode*2.))
    sigma2 = sigma**2
    p = (mode / x) * ops.exp(.5*sigma2 - .5 *((ops.log(x/mode) - sigma2)**2)/sigma2)

    return p

def von_mises_pdf(x, mu, kappa):
    from .backend import bessel_i0
    TWO_PI = 2 * np.pi
    pdf = ops.exp(kappa * ops.cos(x - mu)) / (TWO_PI * bessel_i0(kappa))
    return pdf

# Aggressive softplus with alpha=100
alpha = 100
aggressive_softplus = lambda x: (1./alpha) * ops.softplus(alpha*x)
aggressive_softplus_inverse = lambda y: (1./alpha) * softplus_inverse(alpha * y)


def get_expected_value(stimulus_pdf, normalize=True):

    x = stimulus_pdf.columns.astype(np.float32)

    if normalize:
        stimulus_pdf /= np.trapz(stimulus_pdf, x=x, axis=1)[:, np.newaxis]


    E = np.trapz(stimulus_pdf * x, x=x, axis=1)

    return pd.Series(E, name='E', index=stimulus_pdf.index)


def get_sd_posterior(stimulus_pdf, E=None, normalize=True):

    x = stimulus_pdf.columns.astype(np.float32).values

    if normalize:
        stimulus_pdf /= np.trapz(stimulus_pdf, x=x, axis=1)[:, np.newaxis]

    if E is None:
        E = get_expected_value(stimulus_pdf, normalize=normalize).values
    else:
        if hasattr(E, 'values'):
            E = E.values

    sd = np.sqrt(np.trapz(stimulus_pdf * (x[np.newaxis, :] - E[:, np.newaxis]) ** 2, x=x, axis=1))

    return pd.Series(sd, name='sd', index=stimulus_pdf.index)
