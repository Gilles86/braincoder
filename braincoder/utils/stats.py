import numpy as np


def get_map(p):
    stimuli = p.columns.to_frame(index=False).T
    return stimuli.groupby(level=0).apply(lambda d: (p * d.values).sum(1) / p.sum(1)).T


def get_rsq(data, predictions, zerovartonan=True):

    resid = data - predictions

    ssq_data = np.clip(((data - data.mean(0))**2).sum(0), 1e-5, None)
    ssq_resid = np.clip((resid**2).sum(0), 1e-5, None)

    r2 = (1 - (ssq_resid / ssq_data))

    if zerovartonan:
        r2[data.var() == 0] = np.nan

    return r2
