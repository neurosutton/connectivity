import pandas as pd
import numpy as np
import fmri_analysis_get_data as get
from importlib import reload

def filter_df(df,criteria={}):
    df.dropna(how='all', inplace=True)
    # Other criteria as we build out the pipelines
    return df

def roiIx_to_name_translator():
    faload.load_network_parcels()

def drop_negatives(matrix):
    matrix[matrix < 0] = np.nan
    return matrix

def check_data_loaded():
    import shared
    if not hasattr(shared,'group1_indices'):
        get.get_subj_df_data()
        reload(shared)

def nan_bouncer(x, axis=0):
    # https://stackoverflow.com/questions/48101388/remove-nans-in-multidimensional-array
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
    mask = np.isnan(x)
    cut = np.min(np.count_nonzero(mask, axis=0))
    idx = tuple(np.ogrid[tuple(map(slice, x.shape[1:]))])
    res = x[(np.argsort(~mask, axis=0, kind='mergesort')[cut:],) + idx] 
    return res if axis == 0 else np.moveaxis(res, 0, axis)