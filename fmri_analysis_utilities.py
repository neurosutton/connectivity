import pandas as pd

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
    import fmri_analysis_load_funcs as faload
    shared = faload.load_shared()
    if not hasattr(shared,'conn_data'):
        get_mdata()
        get_conn_data()