import pandas as pd
import numpy as np


def get_rsq(data, predictions):

    data = pd.DataFrame(data)
    predictions = pd.DataFrame(predictions)

    residuals = data - predictions

    return (1 - (residuals.var() / data.var())).T

def get_r(data, predictions):
    """
    Correlation coefficient
    """

    data = pd.DataFrame(data)
    predictions = pd.DataFrame(predictions)
    
    data_ = data - data.mean(0)
    predictions_ = predictions - predictions.mean(0)

    return (data_ * predictions_).sum(0) / np.sqrt((data_**2).sum(0) * (predictions_**2).sum(0))
