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


class shared():
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

def load_shared():
     with open(pkl_file, 'rb') as f:
        shared = pickle.load(f)
     return shared

def load_mat(proj_dir, conn_file):
    """Loading and reloading the module is much quicker with loading the matrix as its own method. Call first, so that there is data, though."""
    mdata = loadmat(os.path.join(proj_dir, conn_file))
    rois = mdata['names']
    rois = [roi[0].replace('hcp_atlas.','') for r_array in rois for roi in r_array]
    return mdata, rois


def get_network_parcels(mdata, network_name):
    """Returns parcel names and indices with HCP remaining in the name and indexed to work with numpy-based functions."""
    parcel_names = [str[0] for str in mdata['names'][0]]
    parcels = {k:v for v,k in enumerate([str[0] for str in mdata['names'][0]])}
    pattern = 'hcp_atlas.' + network_name + '*'
    matching = fnmatch.filter(parcels.keys(), pattern)
    network_parcels = {k:v for k,v in parcels.items() if k in matching}
    indices = [parcels.get(key) for key in matching]
    return network_parcels


def get_parcel_dict(mdata, network_name=None, inverse=False):
    """Alternate method to get ROI indices and names."""
    try:
        parcel_names = [str[0] for str in mdata['names'][0]]
    except:
        print(f'Issue in get_parcel_dict(mdata,network_name={network_name}, inverse={inverse})')
        print(mdata['names'])

    parcel_dict = OrderedDict()
#    if network_name:
#        print(f'Selecting ROIs belonging to the {network_name} network.\n')
    for p,parcel in enumerate(parcel_names):
        parcel = parcel.replace('hcp_atlas.','') # Clean the names
        if network_name and ('whole' not in network_name):
            if network_name.lower() in parcel.lower():
                parcel_dict[parcel] = p
            else:
                pass
                #print(f'Did not find {network_name.lower()} in {parcel.lower()}')
        else:
            parcel_dict[parcel] = p
    if inverse:
        parcel_dict = {v:k for k,v in parcel_dict.items()}
    return parcel_dict

def get_subj_df_data(nonimaging_subjectlevel_data):
    """Primarily for reading in demographic and neuropsychological data."""
    shared = load_shared()
    subj_df = pd.DataFrame(pd.read_csv(nonimaging_subjectlevel_data))
    subj_dict = {k:v for k,v in enumerate(subj_df[shared.name_id_col])}
    group_dict = dict(zip(subj_df[shared.name_id_col], subj_df[shared.group_id_col]))
    return subj_df, subj_dict, group_dict