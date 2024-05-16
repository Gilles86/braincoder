import numpy as np


def get_map(p):
    stimuli = p.columns.to_frame(index=False).T
    return stimuli.groupby(level=0).apply(lambda d: (p * d.values).sum(1) / p.sum(1)).T

def get_rsq(data, predictions, zerovartonan=True, allow_biased_residuals=False):

    resid = data - predictions

    # ssq_data = np.clip(((data - data.mean(0))**2).sum(0), 1e-5, None)
    ssq_data = ((data - data.mean(0))**2).sum(0)
    if allow_biased_residuals:
        ssq_resid = ((resid - resid.mean(0))**2).sum(0)
    else:
        ssq_resid = (resid**2).sum(0)

    r2 = (1 - (ssq_resid / ssq_data))

    if zerovartonan:
        r2[data.var() == 0] = np.nan

    r2.name = 'r2'

    return r2


def get_r(data, predictions):

    data_ = data - data.mean(0)
    predictions_ = predictions - predictions.mean(0)

    r = (data_*predictions_ ).sum(0)
    r = r / (np.sqrt((data_**2).sum(0) * (predictions_**2).sum(0)))

    return r
