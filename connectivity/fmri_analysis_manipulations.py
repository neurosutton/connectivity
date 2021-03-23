"""
Load data from CONN output and create shared variables for other modules in package.

Date: December 2020
@author: Brianne Sutton, PhD from Josh Bear, MD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""

from scipy.io import loadmat
import json
import warnings
import numpy as np
import pandas as pd
from math import ceil, floor
from importlib import reload
from scipy import stats


import fmri_analysis_load_funcs as faload
import fmri_analysis_get_data as get
import fmri_analysis_utilities as utils
import shared
import nia_stats_and_summaries as nia


def calc_mean_conn_data(
        mdata=None,
        roi_count=None,
        clear_triu=True,
        subset=None,
        normalize=False):
    """
    Summarize the connectivity matrix for each included subject (all, 
    if no subset is provided).
    Parameters
    ----------
    mdata: np.array 
        connectivities, imported directly from CONN results
    roi_count: 
    clear_triu : boolean
        Deletes the upper triangle. Does not affect means, but can
        have advserse affects on other measures or representations.
    subset: list
        Subject indices to include in the calculation.

    Returns
    -------
    nxn np.array of means for each ROI by given population
    """

    conn_data = get.get_conn_data(
        mdata=mdata,
        roi_count=roi_count,
        clear_triu=clear_triu,
        subset=subset)
    return np.nanmean(conn_data, axis=2)

def summarize_mean_conn_values(network_name=None, subj_idx=None, exclude=None):
    """
    Figure out the mean connectivity of various networks based on
    raw input from CONN's first-level connectivity matrices.

    Parameters
    ----------
    network_name : string, can be 'whole_brain' or a canonical network from HCP
    subj_idx : int or list of int, numerical subject indices
    exclude : list of subject indices, optional

    Returns
    -------
    nx4 dataframe summarizes the mean connectivity and mean, normed connectivity
    for the specificied networks.
    """

    conn_data = get.get_conn_data(subset=subj_idx)
    network_name = 'whole_brain' if network_name is None else network_name
    subj_idx = np.arange(conn_data.shape[-1]) if subj_idx is None else [subj_idx]
    if exclude:
        subj_idx =  [subj for subj in subj_idx if subj not in exclude]
    mv_df_list = []
    for subj in subj_idx:
        mean_dict = {}
        mat = get.get_network_matrix(network_name,
                                    [subj])
        mat1 = get.get_network_matrix(network_name,
                                    [subj],
                                    normalize=True) 
        mean_dict[subj] = {'mean_conn':np.nanmean(mat), 
                           'mean_conn_normed':np.nanmean(mat1),
                           'network':network_name}
        mv_df_list.append(pd.DataFrame(mean_dict).T)
    df = pd.concat(mv_df_list).reset_index()  
    df = utils.subject_converter(df, orig_subj_col='index').drop(columns=['index'])
    return df


def get_sorted_values(
        mean_conn_data=None,
        mdata=None,
        roi_count=None,
        clear_triu=True):
    """Output: 1D sorted array of connectivity data averaged across population."""
    if mean_conn_data is None:
        mean_conn_data = calc_mean_conn_data(
            mdata=mdata, roi_count=roi_count, clear_triu=clear_triu)
    return np.sort(mean_conn_data[~np.isnan(mean_conn_data)].flatten())


def get_prop_thr_value(
        threshold,
        exclude_negatives=False,
        mean_conn_data=None):
    """Find the index of proportional threshold value based on sorted, population-averaged connectivity at each node."""
    warnings.filterwarnings('ignore')
    mean_conn_data = calc_mean_conn_data() if mean_conn_data is None else mean_conn_data
    sorted_values = get_sorted_values(mean_conn_data=mean_conn_data)
    if exclude_negatives:
        sorted_values = sorted_values[sorted_values >= 0]
    edge_count = len(sorted_values)
    thr_ix = ceil(edge_count * threshold)
    return sorted_values[thr_ix]


def get_edge_count(
        threshold,
        exclude_negatives=False,
        mean_conn_data=None,
        verbose=False):
    """"""
    mean_conn_data = calc_mean_conn_data() if mean_conn_data is None else mean_conn_data
    sorted_values = get_sorted_values(mean_conn_data=mean_conn_data)
    if exclude_negatives:
        sorted_values = sorted_values[sorted_values >= 0]
    edge_count = len(sorted_values)
    if verbose:
        print(
            f'Total number of edges = {edge_count}\nAt {threshold}, {ceil(edge_count * (1-threshold))} edges are needed.')
    return ceil(edge_count * (1 - threshold))


def get_prop_thr_edges(
        threshold,
        exclude_negatives=False,
        mean_conn_data=None,
        subset=[]):
    """Create the mask of nodes greater than the proportional threshold on 
    the population-averaged connectivity. The mask will be applied to 
    individuals' matrices.

    Returns
    -------
    prop_thr_edges : np.array
        binary mask of entire connectivity matrix for an individual
    """
    
    if mean_conn_data is None:
        mean_conn_data = calc_mean_conn_data(subset=subset)
    thr_value = get_prop_thr_value(
        threshold=threshold,
        exclude_negatives=exclude_negatives)
    prop_thr_edges = mean_conn_data.copy()
    prop_thr_edges[prop_thr_edges >= thr_value] = 1
    prop_thr_edges[prop_thr_edges < thr_value] = 0
    return prop_thr_edges


def make_proportional_threshold_mask(
        network_name,
        prop_thr,
        mdata=None,
        exclude_negatives=False,
        subset=[]):
    """Apply population-based, proportional threshold mask to individuals."""
    parcels = get.get_network_parcels(network_name, mdata=mdata)
    indices = list(parcels.values())
    wb_mask = get_prop_thr_edges(
        threshold=prop_thr,
        exclude_negatives=exclude_negatives,
        subset=subset)
    network_mask = wb_mask[np.ix_(indices, indices)]
    return network_mask


def describe_cohort_networks(
        network_name,
        conn_data=None,
        prop_thr=None,
        subject_level=False):
    utils.check_data_loaded()
    conn_data = get.get_conn_data() if conn_data is None else conn_data
    matrix_1 = get.get_cohort_network_matrices(
        network_name,
        shared.group1_indices,
        subject_level=subject_level,
        conn_data=conn_data,
        prop_thr=prop_thr)
    matrix_2 = get.get_cohort_network_matrices(
        network_name,
        shared.group2_indices,
        subject_level=subject_level,
        conn_data=conn_data,
        prop_thr=prop_thr)
    t_test_results = stats.ttest_ind(
        matrix_1, matrix_2, axis=None, nan_policy='omit')
    print(f"Shapes: {matrix_1.shape} | {matrix_2.shape}")
    print(f'Means: {np.nanmean(matrix_1)} | {np.nanmean(matrix_2)}')
    print(f'StDev: {np.nanstd(matrix_1)} | {np.nanstd(matrix_2)}')
    print(f'{t_test_results}')


def prepare_network_edges_statistically(
        network_name, conn_data=None, prop_thr=None):
    """Options for the cohort matrices are intentionally hard coded so that the full connectivity matrix (including the upper triangle) is retained. The full matrix is required for BrainNetViewer to project the edges."""
    utils.check_data_loaded()
    conn_data = get.get_conn_data(
        clear_triu=False) if conn_data is None else conn_data
    matrix_1 = get.get_cohort_network_matrices(
        network_name,
        shared.group1_indices,
        mean=False,
        subject_level=False,
        conn_data=conn_data,
        prop_thr=prop_thr)
    matrix_2 = get.get_cohort_network_matrices(
        network_name,
        shared.group2_indices,
        mean=False,
        subject_level=False,
        conn_data=conn_data,
        prop_thr=prop_thr)
    t_test_results = stats.ttest_ind(
        matrix_1, matrix_2, axis=0, nan_policy='omit')
    print(f"Shapes: {matrix_1.shape} | {matrix_2.shape}")
    print(f'Means: {np.nanmean(matrix_1)} | {np.nanmean(matrix_2)}')
    print(f'StDev: {np.nanstd(matrix_1)} | {np.nanstd(matrix_2)}')
    print(f'{t_test_results}')
    return t_test_results


def get_sig_edges(network_name, mdata=None, prop_thr=None):
    mdata = get.get_mdata() if mdata is None else mdata
    parcels = get.get_network_parcels(network_name, mdata=mdata)
    edges = prepare_network_edges_statistically(
        network_name, prop_thr=prop_thr)
    edges = np.nan_to_num(edges)
    pvals = np.array(edges[1].data)
    pvals_filter = np.where(np.logical_and(pvals < .05, pvals > 0), 1, 0)
    edges = edges[0].data
    edges = edges * pvals_filter
    p_df = pd.DataFrame(
        pvals * pvals_filter,
        columns=parcels.keys(),
        index=parcels.keys())
    results = pd.DataFrame(edges, columns=parcels.keys(), index=parcels.keys())
    results = results.merge(
        p_df,
        left_index=True,
        right_index=True,
        suffixes=[
            '_t',
            '_p'],
        copy=False)
    return edges, results[results != 0].stack()
