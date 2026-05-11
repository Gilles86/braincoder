import numpy as np


def get_map(p):
    stimuli = p.columns.to_frame(index=False).T
    return stimuli.groupby(level=0).apply(lambda d: (p * d.values).sum(1) / p.sum(1)).T

def get_rsq(data, predictions, zerovartonan=True, allow_biased_residuals=False):

    resid = data - predictions

    # Pandas .sum() defaults to skipna=True, which silently drops NaN
    # rows from the sum. When model predictions are NaN (e.g. σ → 0 in
    # softplus collapses the Gaussian / DoG / DN PRF), this returns
    # ssq_resid = 0 and gives the infamous "phantom R² = 1" voxels.
    # Force skipna=False so NaN predictions propagate to NaN R², which
    # downstream code can filter explicitly.
    def _sumna(x):
        try:
            return x.sum(0, skipna=False)
        except TypeError:  # numpy fallback (no skipna kwarg)
            return x.sum(0)

    ssq_data = _sumna((data - data.mean(0)) ** 2)
    if allow_biased_residuals:
        ssq_resid = _sumna((resid - resid.mean(0)) ** 2)
    else:
        ssq_resid = _sumna(resid ** 2)

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
