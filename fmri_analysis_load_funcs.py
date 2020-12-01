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


class config():
    """Provides a single mechanism to pass shared variables around.
    Initial calls to the method should supply the filepath to the definitions file with all the group info, filepaths to directories, etc. in it."""
    def __init__(self,__def_path__=None):
        if __def_path__:
            self.__def_path__ = __def_path__
        with open(os.path.join(self.__def_path__,'directory_defs.json')) as f:
            defs = json.load(f)
            self.conn_dir = defs['conn_dir']
            self.main_dir = defs['main_dir']
            self.atlas_dir = defs['atlas_dir']
            self.proj_dir = defs['data_dir']
            global pkl_file
            pkl_file =  os.path.join(self.proj_dir,'vars.pkl')
            
        self.conn_file = 'resultsROI_Condition001.mat'
        self.nonimaging_subjectlevel_data =  os.path.join(self.main_dir,'eses_subjects_202008.csv')

        self.name_id_col = "BK_name"
        self.group_id_col = "group"
        self.msr_dict = {'cc':"clustering coefficienct", 'pl':"path length",'ms':"mean strength", 'mod':"modularity", 'le':"local efficiency"}
        self.debug = ''

        self.date = datetime.today().strftime('%Y%m')
        self.mdata,x = load_mat(self.proj_dir, os.path.join(self.conn_dir,self.conn_file))
        self.prep_pickle()
     
    def prep_pickle(self):
         with open(pkl_file,'wb') as f:
            pickle.dump(self,f)

def load_config():
     with open(pkl_file, 'rb') as f:
        cfg = pickle.load(f)
     return cfg


def get_mdata():
    return loadmat(op.join(config.conn_dir, config.conn_file))


def load_mat(proj_dir, conn_file):
    """Loading and reloading the module is much quicker with loading the matrix as its own method. Call first, so that there is data, though."""
    mdata = loadmat(os.path.join(proj_dir, conn_file))
    rois = mdata['names']
    rois = [roi[0].replace('hcp_atlas.','') for r_array in rois for roi in r_array]
    return mdata, rois


def get_conn_data(mdata=None, conn_dir=conn_dir, conn_file=conn_file, roi_count=None, clear_triu=True):
    mdata = get_mdata(conn_dir=conn_dir, conn_file=conn_file) if mdata is None else mdata
    roi_count = mdata['Z'].shape[0] if roi_count is None else roi_count
    conn_data = mdata['Z'][:roi_count, :roi_count, :]
    if clear_triu is False:
        return conn_data
    else:
        for subject in range(conn_data.shape[2]):
            conn_data[:, :, subject][np.triu_indices(conn_data.shape[0], 0)] = np.nan
    return conn_data

def get_network_parcels(network_name, subj_idx=None, mdata=None):
    """Returns parcel names and indices with HCP remaining in the name and indexed to work with numpy-based functions."""
    subj_idx = 0 if subj_idx is None else subj_idx
    mdata = tan.get_mdata() if mdata is None else mdata
    parcel_names = [str[0] for str in mdata['names'][0]]
    parcels = {k:v for v,k in enumerate([str[0] for str in mdata['names'][0]])}
    pattern = 'hcp_atlas.' + network_name + '*'
    matching = fnmatch.filter(parcels.keys(), pattern)
    network_parcels = {k:v for k,v in parcels.items() if k in matching}
    indices = [parcels.get(key) for key in matching]
    return network_parcels

def get_subj_df_data(nonimaging_subjectlevel_data):
    """Primarily for reading in demographic and neuropsychological data."""
    cfg = load_config()
    subj_df = pd.DataFrame(pd.read_csv(nonimaging_subjectlevel_data))
    subj_dict = {k:v for k,v in enumerate(subj_df[cfg.name_id_col])}
    group_dict = dict(zip(subj_df[cfg.name_id_col], subj_df[cfg.group_id_col]))
    return subj_df, subj_dict, group_dict