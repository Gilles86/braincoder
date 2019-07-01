import pandas as pd


def get_rsq(data, predictions):

    data = pd.DataFrame(data)
    predictions = pd.DataFrame(predictions)

    residuals = data - predictions

    return 1 - (residuals.var() / data.var())
