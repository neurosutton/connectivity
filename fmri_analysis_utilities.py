import pandas as pd
import fmri_analysis_get_data as get

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
    if not hasattr(shared,'conn_data'):
        get.get_mdata()
        get.get_conn_data()