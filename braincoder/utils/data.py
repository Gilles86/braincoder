import pkg_resources
import numpy as np
import pandas as pd
from scipy import ndimage

def load_szinte2024(resize_factor=1.):

    data = {}

    stream = pkg_resources.resource_stream(__name__, '../data/szinte2024/dm_gazecenter.npz')
    data['stimulus'] = np.load(stream)['arr_0'].astype(np.float32)
    data['grid_coordinates'] = pd.read_csv(pkg_resources.resource_stream(__name__, '../data/szinte2024/grid_coordinates.tsv'), sep='\t').astype(np.float32)

    if resize_factor != 1.:
        data['stimulus'] = np.array([ndimage.zoom(d, 1./resize_factor) for d in data['stimulus']]).astype(np.float32)

        tmp_x = data['grid_coordinates'].set_index(pd.MultiIndex.from_frame(data['grid_coordinates'] ))['x']
        tmp_y = data['grid_coordinates'].set_index(pd.MultiIndex.from_frame(data['grid_coordinates'] ))['y']

        new_x = ndimage.zoom(tmp_x.unstack('y'), 1./resize_factor)
        new_y = ndimage.zoom(tmp_y.unstack('y'), 1./resize_factor)

        # y needs to be flipped because of how stack and ravel works...
        data['grid_coordinates'] = pd.DataFrame({'x':new_x.ravel(), 'y':new_y.ravel()[::-1]}).astype(np.float32)

    data['v1_timeseries'] = pd.read_csv(pkg_resources.resource_stream(__name__, '../data/szinte2024/mask-v1_bold.tsv.gz'), sep='\t', index_col=['time'], compression='gzip').astype(np.float32)
    data['tr'] = 1.317025

    return data