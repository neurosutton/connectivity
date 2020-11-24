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


with open(op.join(op.dirname(op.realpath(__file__)),'directory_defs.json')) as f:
    defs = json.load(f)
    conn_dir = defs['conn_dir']
    conn_file = defs['conn_file']

# subjects_file =  op.join(conn_dir, 'eses_subjects_202008.csv')

def get_mdata(conn_dir=conn_dir, conn_file=conn_file):
    return loadmat(op.join(conn_dir, conn_file))


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


def get_mean_conn_data(mdata=None, conn_dir=conn_dir, conn_file=conn_file, roi_count=None, clear_triu=True):
    conn_data = get_conn_data(mdata=mdata, conn_dir=conn_dir, conn_file=conn_file, roi_count=roi_count, clear_triu=clear_triu)
    return np.nanmean(conn_data, axis=2)


def get_sorted_values(conn_data=None, mdata=None, conn_dir=conn_dir, conn_file=conn_file, roi_count=None, clear_triu=True):
    if conn_data is None:
        conn_data = get_mean_conn_data(mdata=mdata, conn_dir=conn_dir, conn_file=conn_file, roi_count=roi_count, clear_triu=clear_triu)
    return np.sort(conn_data[~np.isnan(conn_data)].flatten())


def get_prop_thr_value(threshold, conn_data=None):
    conn_data = get_mean_conn_data() if conn_data is None else conn_data
    sorted_values = get_sorted_values(conn_data=conn_data)
    edge_count = len(sorted_values)
    position = ceil(edge_count * threshold)
    return sorted_values[position]


def get_prop_thr_edges(threshold, conn_data=None):
    if conn_data is None:
        conn_data = get_mean_conn_data()
    thr_value = get_prop_thr_value(threshold=threshold)
    prop_thr_edges = conn_data.copy()
    prop_thr_edges[prop_thr_edges >= thr_value] = 1
    prop_thr_edges[prop_thr_edges < thr_value] = 0
    return prop_thr_edges
