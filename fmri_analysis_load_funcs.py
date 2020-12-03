"""
Functions to explore connectivity matrices generated through CONN.

Date: August 2020
@author: Brianne Sutton, PhD from Josh Bear, MD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""
import os
from glob import glob
import numpy as np
import fnmatch, random, time, pickle
import pandas as pd
import json
from collections import OrderedDict, defaultdict
from datetime import datetime
from scipy.io import loadmat
from importlib import reload

import fmri_analysis_utilities as utils

class shared_variables():
    """Provides a single mechanism to pass shared variables around.
    Initial calls to the method should supply the filepath to the definitions file with all the group info, filepaths to directories, etc. in it."""
    def __init__(self,__def_path__=None):
        if __def_path__:
            self.__def_path__ = __def_path__
        self.date = datetime.today().strftime('%Y%m')
        with open(os.path.join(self.__def_path__,'directory_defs.json')) as f:
            defs = json.load(f)
            self.conn_dir = defs['conn_dir']
            self.main_dir = defs['main_dir']
            self.atlas_dir = defs['atlas_dir']
            self.proj_dir = defs['data_dir']
            global pkl_file
            pkl_file =  os.path.join(self.proj_dir,'vars.pkl')
            self.name_id_col = defs['name_id_col']
            self.group_id_col = defs['group_id_col']
            self.group1 = defs['group1']
            self.group2 = defs['group2']
            
        self.conn_file = 'resultsROI_Condition001.mat'
        self.nonimaging_subjectlevel_data =  os.path.join(self.main_dir,'eses_subjects_202008.csv')
        get_subj_df_data(self.nonimaging_subjectlevel_data)

        self.mdata = load_mdata(self.proj_dir, os.path.join(self.conn_dir,self.conn_file))
        self.conn_data  = load_conn_data(mdata=self.mdata)
        self.prep_pickle()
     
    def prep_pickle(self):
         with open(pkl_file,'wb+') as f:
            pickle.dump(self,f)

def load_shared():
    with open(pkl_file, 'rb') as f:
        shared = pickle.load(f)
    return shared


def load_mdata(conn_dir=None, conn_file=None):
    """Loading and reloading the module is much quicker with loading the matrix as its own method. Call first, so that there is data, though."""
    if not conn_dir:
        conn_dir = shared.conn_dir
        conn_file = shared.conn_file
    return loadmat(os.path.join(conn_dir, conn_file))


def load_conn_data(mdata=None, roi_count=None, clear_triu=True):
    """Foundational method to transform the MATLAB matrices to numpy matrices.
    Output:
    mdata loaded in the shared object; dictionary of arrays in line with MATLAB data structures.
    conn_data: Just the connectivity matrices extracted from mdata; square matrix excludes the regressors and atlas-based values that CONN adds to the right side of the matrix
    """
    mdata = shared.mdata if mdata is None else mdata
    roi_count = mdata['Z'].shape[0] if roi_count is None else roi_count
    conn_data = mdata['Z'][:roi_count, :roi_count, :]
    if clear_triu is False:
        return conn_data
    else:
        for subject in range(conn_data.shape[2]):
            conn_data[:, :, subject][np.triu_indices(conn_data.shape[0], 0)] = np.nan
    return conn_data

def load_network_parcels(network_name, mdata=None):
    """Returns parcel names and indices with HCP remaining in the name and indexed to work with numpy-based functions.
    Output: {atlas_name.roi: numpy index of ROI}
    """
    mdata = shared.mdata if mdata is None else mdata
    parcel_names = [str[0].lower() for str in mdata['names'][0]]
    parcels = {k:v for v,k in enumerate(parcel_names)}
    pattern = 'hcp_atlas.' + network_name.lower() + '*'
    matching = fnmatch.filter(parcels.keys(), pattern)
    network_parcels = {k:v for k,v in parcels.items() if k in matching}
    #indices = [parcels.get(key) for key in matching] #Unused?
    return network_parcels

def get_subj_df_data(nonimaging_subjectlevel_data=None):
    """Primarily for reading in demographic and neuropsychological data."""
    shared = load_shared()
    nonimaging_subjectlevel_data = shared.nonimaging_subjectlevel_data if nonimaging_subjectlevel_data is None else nonimaging_subjectlevel_data
    subj_df = pd.DataFrame(pd.read_csv(nonimaging_subjectlevel_data))
    subj_df = utils.filter_df(subj_df)
    subj_dict = {k:v for k,v in enumerate(subj_df[shared.name_id_col])}
    group_dict = dict(zip(subj_df[shared.name_id_col], subj_df[shared.group_id_col]))
    grp1_indices = [i for i, x in enumerate(list(subj_df[shared.group_id_col])) if x == shared.group1]
    grp2_indices = [i for i, x in enumerate(list(subj_df[shared.group_id_col])) if x == shared.group2]
    setattr(shared,'group1_indices',grp1_indices)
    setattr(shared,'group2_indices',grp2_indices)
    return subj_df, subj_dict, group_dict

