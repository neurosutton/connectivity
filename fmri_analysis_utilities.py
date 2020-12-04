import pandas as pd
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