"""
Functions to explore connectivity matrices generated through CONN.

Date: August 2020
@author: Josh Bear, MD
v0.2 (BMS) Adapted fmri_analysis_functions for pandas
"""
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
import fnmatch
from matplotlib import colors
from matplotlib.pyplot import figure
import pandas as pd
from matplotlib import cm
import json
import seaborn as sns
from scipy.stats import pearsonr
import fmri_analysis_get_data as get
import fmri_analysis_load_funcs as faload
import fmri_analysis_manipulations as fam
import shared


# Change log
# 03.11.21 BMS
# Removed equals signs in the printed statements of describe_cohort_networks that broke imports

def get_network_matrix(
        network_name,
        subj_idx,
        conn_data=None,
        mdata=None,
        prop_thr=None,
        network_mask=None,
        exclude_negatives=False,
        normalize=False):
    '''
    Adding a normalize, which can call different types.
        - 'self' will divide by own whole brain mean connectivity
    '''
    prop_thr is None if prop_thr == 0 else prop_thr
    conn_data = get.get_conn_data() if conn_data is None else conn_data
    parcels = get.get_network_parcels(network_name, mdata=mdata)
    indices = list(parcels.values())
    matrix = conn_data[:, :, subj_idx][np.ix_(indices, indices)]
    if prop_thr or network_mask is not None:
        if prop_thr:
            network_mask = get_proportional_threshold_mask(
                network_name=network_name,
                prop_thr=prop_thr,
                conn_data=conn_data,
                mdata=mdata,
                exclude_negatives=exclude_negatives)
        matrix = network_mask * matrix
        matrix[matrix == 0] = np.nan
    if normalize is not False:
        # for start, will just assume it's 'self'
        self_norm_value = np.nanmean(
            tan.drop_negatives(conn_data[:, :, subj_idx]))
        matrix = matrix / np.absolute(self_norm_value)
    return matrix


def get_cohort_network_matrices(
        network_name,
        subj_idx,
        mean=False,
        conn_data=None,
        prop_thr=None,
        subject_level=False,
        network_mask=None,
        exclude_negatives=False,
        normalize=False  ):
    """
    Parameters
    ----------
    
    Returns
    -------

    """

    conn_data = get.get_conn_data() if conn_data is None else conn_data
    ''' Get the matrices for a cohort of patients in a given network. '''
    cohort_matrices = []  # need to collect all the matrices to add
    for subj in subj_idx:
        matrix = get_network_matrix(
            network_name,
            subj,
            conn_data=conn_data,
            prop_thr=prop_thr,
            network_mask=network_mask,
            exclude_negatives=exclude_negatives)
        cohort_matrices.append(matrix)
    cohort_matrices = np.asarray(cohort_matrices)
    if mean is True:
        return np.nanmean(cohort_matrices, axis=0)
    elif subject_level is True:
        return np.nanmean(cohort_matrices, axis=(1, 2))
    else:
        return cohort_matrices


def describe_cohort_networks(
        network_name,
        subj_idx_list_1,
        subj_idx_list_2,
        conn_data=None,
        prop_thr=None,
        subject_level=False):
    """
    Parameters
    ----------

    Returns
    -------
    Prints comparison statistics.
    """

    conn_data = get.get_conn_data() if conn_data is None else conn_data
    matrix_1 = get_cohort_network_matrices(
        network_name,
        subj_idx_list_1,
        subject_level=subject_level,
        conn_data=conn_data,
        prop_thr=prop_thr)
    matrix_2 = get_cohort_network_matrices(
        network_name,
        subj_idx_list_2,
        subject_level=subject_level,
        conn_data=conn_data,
        prop_thr=prop_thr)
    t_test_results = scipy.stats.ttest_ind(
        matrix_1, matrix_2, axis=None, nan_policy='omit')
    print(f'Shapes: {matrix_1.shape} | {matrix_2.shape}')
    print(f'Means: {np.nanmean(matrix_1)} | {np.nanmean(matrix_2)}')
    print(f'StDev: {np.nanstd(matrix_1)} | {np.nanstd(matrix_2)}')
    print(f'{t_test_results}')


def get_cohort_comparison_over_thresholds(
        network_name,
        group_indices,
        group_names=None,
        thr_range=None,
        thr_increment=None,
        conn_data=None,
        subject_level=False,
        plot=False,
        exclude_negatives=False):
    """
    Parameters
    ----------
    group_indices : list of lists

    Returns
    -------
    comp_df : 
    """

    conn_data = get.get_conn_data() if conn_data is None else conn_data
    thr_increment = 0.1 if thr_increment is None else thr_increment
    thr_range = np.arange(
        0., 1, thr_increment) if thr_range is None else thr_range
    group_names = ['1', '2'] if group_names is None else group_names
    comp_df = pd.DataFrame(columns=['threshold', 'group', 'connectivity'])
    df_idx = 0
    for value in thr_range:
        network_mask = fam.make_proportional_threshold_mask(
            network_name, value, exclude_negatives=exclude_negatives)
        matrix_1 = get_cohort_network_matrices(
            network_name,
            group_indices[0],
            subject_level=subject_level,
            conn_data=conn_data,
            network_mask=network_mask)
        matrix_2 = get_cohort_network_matrices(
            network_name,
            group_indices[1],
            subject_level=subject_level,
            conn_data=conn_data,
            network_mask=network_mask)
        for conn in matrix_1.flatten():
            if not np.isnan(conn):
                comp_df.loc[df_idx] = [value, group_names[0], conn]
                df_idx = df_idx + 1
        for conn in matrix_2.flatten():
            if not np.isnan(conn):
                comp_df.loc[df_idx] = [value, group_names[1], conn]
                df_idx = df_idx + 1
    # fixes a potential rounding error in np.arange
    comp_df = comp_df.round(decimals={'threshold': 2})
    if plot:
        plot_cohort_comparison_over_thresholds(
            network_name, comp_df, group_names)
    return comp_df


def get_subject_scores(measure):
    """
    Parameters
    ----------
    measure: str
        Graph measure column header

    Returns
    -------
    scores_df : DataFrame
        Subsetted dataframe that contains the subject labels and scores for the 
        graph measure of interest.
    """
    
    scores = {}
    scores_df = pd.DataFrame(columns=['index', 'subject', measure])

    for row in subj_data.index:
        if not np.isnan(float(subj_data[subj_data.index == row][measure])):
            scores_df.loc[len(scores_df)] = [row,
                                             str(subj_data[subj_data.index == row]['subject'].values[0]),
                                             float(subj_data[subj_data.index == row][measure])]
            # scores[row] = float(subj_data[subj_data.index == row][measure])
    return scores_df
