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

import fmri_analysis_utilities as utils
import fmri_analysis_load_funcs as faload
shared = faload.load_shared()

def test_shared():
    return shared


def get_mdata(conn_dir=None, conn_file=None):
    """Loading and reloading the module is much quicker with loading the matrix as its own method. Call first, so that there is data, though."""
    if not conn_dir:
        conn_dir = shared.conn_dir
        conn_file = shared.conn_file
    shared.mdata = loadmat(os.path.join(conn_dir, conn_file))
    return loadmat(os.path.join(conn_dir, conn_file))


def get_conn_data(mdata=None, roi_count=None, clear_triu=True):
    """Foundational method to transform the MATLAB matrices to numpy matrices.
    Output:
    mdata loaded in the shared object; dictionary of arrays in line with MATLAB data structures.
    conn_data: Just the connectivity matrices extracted from mdata; square matrix excludes the regressors and atlas-based values that CONN adds to the right side of the matrix
    """
    mdata = shared.mdata if mdata is None else mdata
    roi_count = mdata['Z'].shape[0] if roi_count is None else roi_count
    conn_data = mdata['Z'][:roi_count, :roi_count, :]
    if clear_triu:
        for subject in range(conn_data.shape[2]):
            conn_data[:, :, subject][np.triu_indices(conn_data.shape[0], 0)] = np.nan
    shared.conn_data = conn_data
    return conn_data

def get_network_parcels(network_name, mdata=None):
    """Returns parcel names and indices with HCP remaining in the name and indexed to work with numpy-based functions.
    Output: {atlas_name.roi: numpy index of ROI}
    """
    if not hasattr(shared,'mdata'):
        get_mdata()
        get_conn_data()
        faload._pickle(shared)
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
    nonimaging_subjectlevel_data = shared.nonimaging_subjectlevel_data if nonimaging_subjectlevel_data is None else nonimaging_subjectlevel_data
    subj_df = pd.DataFrame(pd.read_csv(nonimaging_subjectlevel_data))
    subj_df = utils.filter_df(subj_df)
    subj_dict = {k:v for k,v in enumerate(subj_df[shared.name_id_col])}
    group_dict = dict(zip(subj_df[shared.name_id_col], subj_df[shared.group_id_col]))
    grp1_indices = [i for i, x in enumerate(list(subj_df[shared.group_id_col])) if x == shared.group1]
    grp2_indices = [i for i, x in enumerate(list(subj_df[shared.group_id_col])) if x == shared.group2]
    shared.group1_indices = grp1_indices
    shared.group2_indices = grp2_indices
    faload._pickle(shared)
    return subj_df, subj_dict, group_dict

