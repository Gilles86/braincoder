from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterOptimizer
import numpy as np
import scipy.stats as ss

def get_correlation_matrix(a, b):

    assert(a.ndim == 2)
    assert(b.ndim == 2)

    if hasattr(a, 'values'):
        a = a.values

    if hasattr(b, 'values'):
        b = b.values

    a = (a - a.mean(0)) / np.var(a)
    b = (b - b.mean(0)) / np.var(b)

    return np.mean(a*b, 0)


def get_paradigm():
    return np.linspace(-5, 5, dtype=np.float32)[:, np.newaxis]


def get_parameters(n_pars=100):
    mus = np.random.rand(n_pars) * 3. - 1.5
    sds = np.random.rand(n_pars)*3
    amplitudes = np.random.rand(n_pars) * 2
    baselines = np.random.rand(n_pars) * 2 - 1

    parameters = np.concatenate((mus[:, np.newaxis],
				 sds[:, np.newaxis],
				 amplitudes[:, np.newaxis],
				 baselines[:, np.newaxis]), 1)

    return parameters

def test_gauss_prf():

    paradigm = get_paradigm()
    parameters = get_parameters()

    model = GaussianPRF()

    data = model.simulate(paradigm, parameters, noise=.1)

    optimizer = ParameterOptimizer(model, data, paradigm)

    optimizer.fit()

    corr = get_correlation_matrix(parameters, optimizer.estimated_parameters)
    print(corr)

    assert(np.all(corr > 0.1))


