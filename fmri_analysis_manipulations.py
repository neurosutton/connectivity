"""
Load data from CONN output and create shared variables for other modules in package.

Date: December 2020
@author: Brianne Sutton, PhD from Josh Bear, MD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""

from scipy.io import loadmat
import os.path as op
import json
import numpy as np
from math import ceil, floor


import fmri_analysis_load_funcs as faload
shared = get.get_shared()
import fmri_analysis_get_data as get
import fmri_analysis_utilities as utils


def calc_mean_conn_data(mdata=None, roi_count=None, clear_triu=True):
    conn_data = get.get_conn_data(mdata=mdata, roi_count=roi_count, clear_triu=clear_triu)
    return np.nanmean(conn_data, axis=2)


def get_sorted_values(mean_conn_data=None, mdata=None, roi_count=None, clear_triu=True):
    """Output: 1D sorted array of connectivity data averaged across population."""
    if mean_conn_data is None:
        mean_conn_data = calc_mean_conn_data(mdata=mdata, roi_count=roi_count, clear_triu=clear_triu)
    return np.sort(mean_conn_data[~np.isnan(mean_conn_data)].flatten())


def get_prop_thr_value(threshold, exclude_negatives=False, mean_conn_data=None):
    """Find the index of proportional threshold value based on sorted, population-averaged connectivity at each node."""
    mean_conn_data = calc_mean_conn_data() if mean_conn_data is None else mean_conn_data
    sorted_values = get_sorted_values(mean_conn_data=mean_conn_data)
    if exclude_negatives:
        sorted_values = sorted_values[sorted_values >= 0]
    edge_count = len(sorted_values)
    thr_ix = ceil(edge_count * threshold)
    return sorted_values[thr_ix]

def get_prop_thr_edges(threshold, exclude_negatives=False, mean_conn_data=None):
    """Create the mask of nodes greater than the proportional threshold on the population-averaged connectivity. The mask will be applied to individuals' matrices."""
    if mean_conn_data is None:
        mean_conn_data = calc_mean_conn_data()
    thr_value = get_prop_thr_value(threshold=threshold, exclude_negatives=exclude_negatives)
    prop_thr_edges = mean_conn_data.copy()
    prop_thr_edges[prop_thr_edges >= thr_value] = 1
    prop_thr_edges[prop_thr_edges < thr_value] = 0
    return prop_thr_edges

def make_proportional_threshold_mask(network_name, prop_thr, mdata=None, exclude_negatives=False):
    """Apply population-based, proportional threshold mask to individuals."""
    parcels = get.get_network_parcels(network_name, mdata=mdata)
    indices = list(parcels.values())
    wb_mask = get_prop_thr_edges(threshold=prop_thr, exclude_negatives=exclude_negatives)
    network_mask = wb_mask[np.ix_(indices, indices)]
    return network_mask


def describe_cohort_networks(network_name, conn_data=None, prop_thr=None, subject_level=False):
    utils.check_data_loaded()
    conn_data = shared.conn_data if conn_data is None else conn_data
    matrix_1 = get_cohort_network_matrices(network_name, shared.group1_indices, subject_level=subject_level, conn_data=conn_data, prop_thr=prop_thr)
    matrix_2 = get_cohort_network_matrices(network_name, shared.group2_indices, subject_level=subject_level, conn_data=conn_data, prop_thr=prop_thr)
    t_test_results = scipy.stats.ttest_ind(matrix_1, matrix_2, axis=None, nan_policy='omit')
    print(f'Shapes: {matrix_1.shape=} | {matrix_2.shape=}')
    print(f'Means: {np.nanmean(matrix_1)=} | {np.nanmean(matrix_2)=}')
    print(f'StDev: {np.nanstd(matrix_1)=} | {np.nanstd(matrix_2)=}')
    print(f'{t_test_results=}')