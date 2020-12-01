import os
from glob import glob
import numpy as np
import pandas as pd
import json
from collections import OrderedDict, defaultdict
from datetime import datetime
import multiprocessing
import itertools

import fmri_analysis_utilities as utils
import fmri_analysis_load_funcs as faload
import fmri_analysis_matrices as fam

config = faload.load_config()

def calc_mean_fc(network_name=config.network):
    print('Calculating the mean strength.')
    if config.wb_norm:
        norm='pop'
    else:
        norm='indvd'
    prop_name = str(config.prop_thr).split('.')[-1]
    if not network_name or network_name == 'brain':
        network_name='whole_brain'

    m=fam.analysis()
    graph_df = m.produce_matrix()

    try:
        cols = [col for col in graph_df.columns if 'fc' in col]
        graph_df = graph_df.drop(columns=cols)
    except:
        print(f'issue with {cols}')

    add_df = pd.DataFrame(index=rois, columns=['fc'])
    for subj in set(network_df[name_id_col]):
        tmp = pd.DataFrame(index=rois)
        tmp['fc'] = network_df.loc[network_df[name_id_col]==subj,rois].mean(axis=0)
        tmp[name_id_col] = subj
        add_df = pd.concat([add_df,tmp])
        add_df.dropna(axis=0, subset=['fc'],inplace=True)
    add_df['network'] = network_name
    add_df.reset_index(inplace=True)
    add_df.rename({'index':'rois'}, axis=1, inplace=True)
    graph_df = graph_df.merge(add_df, how='left', on=[name_id_col,'network','rois'],copy=False)
    graph_df.to_csv(graph_df_file,index=False)

    return graph_df