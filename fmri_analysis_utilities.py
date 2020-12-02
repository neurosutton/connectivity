import pandas as pd

def filter_df(df,criteria={}):
    df.dropna(how='all', inplace=True)
    # Other criteria as we build out the pipelines
    return df

def roiIx_to_name_translator():
    faload.load_network_parcels()