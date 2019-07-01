import numpy as np
import scipy as sp
import scipy.stats as ss
from scipy import integrate
from .basis_functions import gaussian_normed
import pandas as pd


class GaussianEncodingModel(object):

    def __init__(self,
                 means,
                 sds,
                 amplitude=1.0,
                 baseline=0.0,
                 weights=None):
        self.means = np.atleast_2d(means)
        sds = np.atleast_2d(sds)

        if sds.ndim == 0:
            self.sds = np.ones_like(self.means) * sds
        else:
            self.sds = sds

        if weights is None:
            self.W = np.identity(self.means.shape[1])
        else:
            self.W = weights

        self.amplitude = amplitude
        self.baseline = baseline

        self.basis_function = gaussian_normed
        self.max = np.max(self.means + 4 * self.sds)

    def get_response_profile(self, n):
        n = np.atleast_2d(n).T
        return self.basis_function(n, self.means, self.sds)

    def get_bold_distribution(self, n, noise=.1):
        """
        Gives a multivariate normal object that can generate
        data according to the model.
        """

        profile = self.get_response_profile(n)

        multi = ss.multivariate_normal(
            self.W.dot(profile.T).ravel(), cov=noise)

        return multi

    def simulate_data(self, ns, noise=.1):

        dists = []
        for n in ns:
            dists.append(self.get_bold_distribution(n, noise).rvs())

        return pd.DataFrame(dists)

    def get_decoding_dist(self, n, noise=0.1, n_=None):
        """
        Gives a multivariate normal object that can generate
        data according to the model.
        """
        if n_ is None:
            n_ = np.linspace(0, self.max, 1000)

        m = self.get_bold_distribution(n, noise)
        p = m.pdf(self.get_response_profile(n_))

        den = integrate.quad(lambda x: m.pdf(
            self.get_response_profile(x)), 0, 3 * self.max)[0]

        return n_, p / den

    def _get_random_samples(self, n, noise=0.1, n_=None, num_samples=1000):

        n_, p = self.get_decoding_dist(n, noise, n_)
        i = sp.interpolate.interp1d(np.cumsum(p) / p.sum(), n_)

        return i(np.random.rand(num_samples))
