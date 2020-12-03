"""
Functions to explore connectivity matrices generated through CONN.

Date: August 2020
@author: Josh Bear, MD with organization by Brianne Sutton, PhD
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
import fmri_analysis_plotting as faplot

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


def get_subject_scores(measure):
    """Gather cognitive or medical scores."""
    scores = {}
    scores_df = pd.DataFrame(columns=['index', 'subject', measure])
    subj_data,x,x = get_subjget_subj_df_data()

    for row in subj_data.index:
        if not np.isnan(float(subj_data[subj_data.index == row][measure])):
            scores_df.loc[len(scores_df)] = [row,
                                             str(subj_data[subj_data.index == row]['subject'].values[0]),
                                             float(subj_data[subj_data.index == row][measure])]
            # scores[row] = float(subj_data[subj_data.index == row][measure])
    return scores_df

def get_network_matrix(network_name, subj_idx, conn_data=None, prop_thr=None, network_mask=None,
                       exclude_negatives=False, normalize=False):
    '''
    Adding a normalize, which can call different types.
        - 'self' will divide by own whole brain mean connectivity
    '''
    utils.check_data_loaded()
    prop_thr is None if prop_thr == 0 else prop_thr
    conn_data = shared.conn_data if conn_data is None else conn_data
    parcels = get.get_network_parcels(network_name)
    indices = list(parcels.values())
    matrix = conn_data[:, :, subj_idx][np.ix_(indices, indices)]
    if prop_thr or network_mask is not None:
        if prop_thr:
            network_mask = fam.make_proportional_threshold_mask(network_name=network_name,
                                                           prop_thr=prop_thr, conn_data=conn_data,
                                                           exclude_negatives=exclude_negatives)
        matrix = network_mask * matrix
        matrix[matrix == 0] = np.nan
    if normalize is not False:
        # for start, will just assume it's 'self'
        self_norm_value = np.nanmean(utils.drop_negatives(conn_data[:, :, subj_idx]))
        matrix = matrix / np.absolute(self_norm_value)
    return matrix

def get_cohort_network_matrices(network_name, subj_idx, mean=False, conn_data=None, prop_thr=None,
                                subject_level=False, network_mask=None, exclude_negatives=False):
    conn_data = shared.conn_data if conn_data is None else conn_data
    ''' Get the matrices for a cohort of patients in a given network. '''
    cohort_matrices = []  # need to collect all the matrices to add
    for subj in subj_idx:
        matrix = get_network_matrix(network_name, subj, conn_data=conn_data,
                                    prop_thr=prop_thr, network_mask=network_mask,
                                    exclude_negatives=exclude_negatives)
        cohort_matrices.append(matrix)
    cohort_matrices = np.asarray(cohort_matrices)
    if mean is True:
        return np.nanmean(cohort_matrices, axis=0)
    elif subject_level is True:
        return np.nanmean(cohort_matrices, axis=(1, 2))
    else:
        return cohort_matrices

def get_cohort_comparison_over_thresholds(network_name, group_indices, thr_range=None, thr_increment=None, conn_data=None, subject_level=False, plot=False, exclude_negatives=False):
    conn_data = shared.conn_data if conn_data is None else conn_data
    thr_increment = 0.1 if thr_increment is None else thr_increment
    thr_range = np.arange(0., 1, thr_increment) if thr_range is None else thr_range
    group_names = [shared.group1, shared.group2]
    comp_df = pd.DataFrame(columns=['threshold', 'group', 'connectivity'])
    df_idx = 0
    for value in thr_range:
        network_mask = make_proportional_threshold_mask(network_name, value, exclude_negatives=exclude_negatives)
        matrix_1 = get_cohort_network_matrices(network_name, shared.group1_indices, subject_level=subject_level, conn_data=conn_data, network_mask=network_mask)
        matrix_2 = get_cohort_network_matrices(network_name, shared.group2_indices, subject_level=subject_level, conn_data=conn_data, network_mask=network_mask)
        for conn in matrix_1.flatten():
            if not np.isnan(conn):
                comp_df.loc[df_idx] = [value, group_names[0], conn]
                df_idx = df_idx + 1
        for conn in matrix_2.flatten():
            if not np.isnan(conn):
                comp_df.loc[df_idx] = [value, group_names[1], conn]
                df_idx = df_idx + 1
    comp_df = comp_df.round(decimals={'threshold': 2})  # fixes a potential rounding error in np.arange
    if plot:
        faplot.plot_cohort_comparison_over_thresholds(network_name, comp_df, group_names)
    return comp_df