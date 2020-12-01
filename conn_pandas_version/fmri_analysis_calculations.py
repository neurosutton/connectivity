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

def calc_mean_fc(network_name=None):
    print('Calculating the mean strength.')
    if config.wb_norm:
        norm='pop'
    else:
        norm='indvd'
    prop_name = str(config.prop_thr).split('.')[-1]
    if not network_name or network_name == 'brain':
        network_name='whole_brain'

    m=fam.analysis()
    network_df = m.produce_matrix()

    subj_dfs = []
    for subj in set(network_df[config.name_id_col]):
        tmp = pd.DataFrame(index=rois)
        tmp['fc'] = network_df.loc[network_df[config.name_id_col]==subj,rois].mean(axis=0)
        tmp[name_id_col] = subj
        subj_dfs.append(tmp)
    mean_df = pd.concat(add_dfs)
    mean_df['network'] = network_name
    mean_df.reset_index(inplace=True)
    mean_df.rename({'index':'rois'}, axis=1, inplace=True)

    return add_df