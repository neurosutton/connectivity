"""
Functions to threshold and normalize connectivity matrices generated through CONN.

Date: November 2020
@author: Joshua J Bear, MD, MA
"""

from scipy.io import loadmat
import os.path as op
import json
import numpy as np
from math import ceil, floor


import fmri_analysis_load_funcs as faload

shared = faload.load_shared()

# Moved to load_funcs:
# get_mdata
# get_conn_data


def test_shared_availability():
    return shared.group_id_col


def get_mean_conn_data(mdata=None, roi_count=None, clear_triu=True):
    conn_data = faload.load_conn_data(
        mdata=mdata,
        roi_count=roi_count,
        clear_triu=clear_triu)
    return np.nanmean(conn_data, axis=2)


def get_sorted_values(
        conn_data=None,
        mdata=None,
        roi_count=None,
        clear_triu=True):
    if conn_data is None:
        conn_data = get_mean_conn_data(
            mdata=mdata,
            roi_count=roi_count,
            clear_triu=clear_triu)
    return np.sort(conn_data[~np.isnan(conn_data)].flatten())


def get_prop_thr_value(threshold, exclude_negatives=False, conn_data=None):
    conn_data = get_mean_conn_data() if conn_data is None else conn_data
    sorted_values = get_sorted_values(conn_data=conn_data)
    if exclude_negatives:
        sorted_values = sorted_values[sorted_values >= 0]
    edge_count = len(sorted_values)
    thr_ix = ceil(edge_count * threshold)
    return sorted_values[thr_ix]


def get_proportional_threshold_mask(
        network_name,
        prop_thr,
        subj_idx=None,
        conn_data=None,
        mdata=None,
        exclude_negatives=False):
    # conn_data = tan.get_conn_data() if conn_data is None else conn_data
    parcels = faload.load_network_parcels(
        network_name, subj_idx=subj_idx, mdata=mdata)
    indices = list(parcels.values())
    wb_mask = tan.get_prop_thr_edges(
        threshold=prop_thr,
        exclude_negatives=exclude_negatives)
    network_mask = wb_mask[np.ix_(indices, indices)]
    return network_mask


def get_prop_thr_edges(threshold, exclude_negatives=False, conn_data=None):
    if conn_data is None:
        conn_data = get_mean_conn_data()
    thr_value = get_prop_thr_value(
        threshold=threshold,
        exclude_negatives=exclude_negatives)
    prop_thr_edges = conn_data.copy()
    prop_thr_edges[prop_thr_edges >= thr_value] = 1
    prop_thr_edges[prop_thr_edges < thr_value] = 0
    return prop_thr_edges


def drop_negatives(matrix):
    matrix[matrix < 0] = np.nan
    return matrix
