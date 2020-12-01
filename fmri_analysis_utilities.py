
import pandas as pd
import numpy as np
import fmri_analysis_load_funcs as faload

config = faload.load_config()

def filter_conn_df_network(df, rois):
    # Limits the df to the chosen network
    cols = [col for col in df.columns if col in ([config.name_id_col, config.group_id_col] + rois)]
    conn_df = df[cols][df.index.isin(rois)]
    return conn_df

def filter_conn_df_subjects(df, subj_list):
    if subj_list:
        if not isinstance(subj_list,list):
            subj_list = [subj_list] # Is a single person was specified.
    print(f'Gathering {subj_list}')
    if isinstance(subj_list[0], int):
        conn_df = df.iloc[subj_list,:] # Compatibility with numpy logic
    else:
        conn_df = df.loc[conn_df[config.name_id_col].isin(subj_list),:]
    return conn_df

def _abs_val_thr(df, prop_thr):
    rois = set(df.index)
    tmp = df.loc[df[rois, rois]].abs().to_numpy(na_value=0)
    sign_mask = np.sign(tmp)
    tmp = _thr(tmp,prop_thr)
    df1 = pd.DataFrame(data=tmp, index = rois, columns=rois)
    df1 = df1*sign_mask
    df1 = df1.replace({-0:np.nan})
    df.loc[df[rois, rois]] = df1
    return df

def _threshold(array, prop_thr):
    if not isinstance(array, np.ndarray):
        array = array.to_numpy(na_value=0)
    return bct.threshold_proportional(array, prop_thr, copy=False)

def _triu(df):
    rois = list(set(df.index))
    if len(set(df[config.name_id_col])) > 1:
        for s in set(df[config.name_id_col]):
            cols = [config.name_id_col] + rois
            tmp = df.loc[df[config.name_id_col]==s,[cols]]
            tmp[np.triu_indices(tmp.shape[0], k=0)] = np.nan
            df.loc[df[config.name_id_col]==s,cols] = tmp
    else:
        tmp = df.loc[df[:,rois]]
        tmp[np.triu_indices(tmp.shape[0], k=0)] = np.nan
        df.loc[df[:,rois]] = tmp
    return df

def _lowercase(input_list):
    return [str(el).lower() for el in input_list]

def _normalize(df):
    rois = list(set(df.index))
    (x, y) = df[rois].shape
    mat = df[rois].to_numpy(na_value=0)
    mat = mat*(mat>0)
    np.fill_diagonal(mat,0) # BCT compatibility
    df[rois] = (mat - mat.min())/(mat.max()-mat.min()) # Since the scaling is (0,1), there is no need to multiply by anything else
    df[rois] = df[rois].replace({0:np.nan})
    return df

def _mean_matrix(df):
    return(df[set(df.index)].mean().mean())