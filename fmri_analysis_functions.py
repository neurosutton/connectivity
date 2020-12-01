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
import thresholding_and_normalizing as tan
import seaborn as sns
from scipy.stats import pearsonr


with open(op.join(op.dirname(op.realpath(__file__)),'directory_defs.json')) as f:
    defs = json.load(f)
    conn_dir = defs['conn_dir']
    conn_file = defs['conn_file']
    subjects_file = defs['subjects_file']

# subject_file = op.join(conn_dir,'eses_subjects_202008.csv')

subj_data = pd.read_csv(op.join(conn_dir, subjects_file))
eses_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'eses']
hc_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'hc']


def get_network_parcels(network_name, subj_idx=None, mdata=None):
    subj_idx = 0 if subj_idx is None else subj_idx
    mdata = tan.get_mdata() if mdata is None else mdata
    parcel_names = [str[0] for str in mdata['names'][0]]
    parcels = {k:v for v,k in enumerate([str[0] for str in mdata['names'][0]])}
    pattern = 'hcp_atlas.' + network_name + '*'
    matching = fnmatch.filter(parcels.keys(), pattern)
    network_parcels = {k:v for k,v in parcels.items() if k in matching}
    indices = [parcels.get(key) for key in matching]
    return network_parcels


def get_network_matrix(network_name, subj_idx, conn_data=None, mdata=None, prop_thr=None, network_mask=None,
                       exclude_negatives=False, normalize=False):
    '''
    Adding a normalize, which can call different types.
        - 'self' will divide by own whole brain mean connectivity
    '''
    prop_thr is None if prop_thr == 0 else prop_thr
    conn_data = tan.get_conn_data() if conn_data is None else conn_data
    parcels = get_network_parcels(network_name, subj_idx, mdata=mdata)
    indices = list(parcels.values())
    matrix = conn_data[:, :, subj_idx][np.ix_(indices, indices)]
    if prop_thr or network_mask is not None:
        if prop_thr:
            network_mask = get_proportional_threshold_mask(network_name=network_name,
                                                           prop_thr=prop_thr, conn_data=conn_data,
                                                           mdata=mdata, exclude_negatives=exclude_negatives)
        matrix = network_mask * matrix
        matrix[matrix == 0] = np.nan
    if normalize is not False:
        # for start, will just assume it's 'self'
        self_norm_value = np.nanmean(tan.drop_negatives(conn_data[:, :, subj_idx]))
        matrix = matrix / np.absolute(self_norm_value)
    return matrix


def get_proportional_threshold_mask(network_name, prop_thr, subj_idx=None, conn_data=None,
                                    mdata=None, exclude_negatives=False):
    # conn_data = tan.get_conn_data() if conn_data is None else conn_data
    parcels = get_network_parcels(network_name, subj_idx=subj_idx, mdata=mdata)
    indices = list(parcels.values())
    wb_mask = tan.get_prop_thr_edges(threshold=prop_thr, exclude_negatives=exclude_negatives)
    network_mask = wb_mask[np.ix_(indices, indices)]
    return network_mask


def get_cohort_network_matrices(network_name, subj_idx, mean=False, conn_data=None, prop_thr=None,
                                subject_level=False, network_mask=None, exclude_negatives=False):
    conn_data = tan.get_conn_data() if conn_data is None else conn_data
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


def plot_network_matrix(conn_data, network_name, subj_idx):
    parcels = get_network_parcels(conn_data, network_name, subj_idx)
    indices = list(parcels.values())
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(conn_data['Z'][:, :, subj_idx][np.ix_(indices, indices)])
    fig.colorbar(im)
    # plt.matshow(conn_data['Z'][:, :, 0][np.ix_(indices, indices)])
    # Let's adjust the tick labels.
    plt.title(f'Subject: {subj_idx} | Network: {network_name}')
    plt.xticks(np.arange(len(indices)), list(parcels.keys()), rotation='vertical')
    plt.yticks(np.arange(len(indices)), list(parcels.keys()), rotation='horizontal')
    # plt.colorbar()
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
    plt.show()


def plot_cohort_network_matrix(conn_data, network_name, subj_idx_list):
    cohort_matrices = []  # need to collect all the matrices to add
    for subj in subj_idx_list:
        cohort_matrices.append(get_network_matrix(mdata, network_name, subj))
    cohort_matrices = np.asarray(cohort_matrices)
    mean_matrix = np.nanmean(cohort_matrices, axis=0)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(mean_matrix)
    fig.colorbar(im)
    plt.show()


def plot_cohort_comparison_matrices(network_name, subj_idx_list_1, subj_idx_list_2, prop_thr=None, vmin=None, vmax=None, conn_data=None, mdata=None):
    conn_data = tan.get_conn_data() if conn_data is None else conn_data
    mean_matrix_1 = get_cohort_network_matrices(network_name, subj_idx_list_1, mean=True, conn_data=conn_data, prop_thr=prop_thr)
    mean_matrix_2 = get_cohort_network_matrices(network_name, subj_idx_list_2, mean=True, conn_data=conn_data, prop_thr=prop_thr)
    vmin = np.min([np.nanmin(mean_matrix_1), np.nanmin(mean_matrix_2)]) if vmin is None else vmin
    vmax = np.max([np.nanmax(mean_matrix_1), np.nanmax(mean_matrix_2)]) if vmax is None else vmax
    boundary = np.max([np.absolute(vmin), np.absolute(vmax)])
    parcels = get_network_parcels(network_name, subj_idx_list_1[0], mdata=mdata)
    indices = list(parcels.values())
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=180)
    cmap = plt.get_cmap('Spectral')
    cNorm = colors.Normalize(vmin=-boundary, vmax=boundary)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    im1 = axs[0].matshow(mean_matrix_1, cmap=cmap, norm=cNorm)
    im2 = axs[1].matshow(mean_matrix_2, cmap=cmap, norm=cNorm)
    if len(mean_matrix_1[0]) < 25:
        axs[0].set_xticklabels(list(parcels.keys()), rotation='vertical', fontsize=5)
        axs[0].set_xticks(np.arange(len(indices)))
        axs[0].set_yticklabels(list(parcels.keys()), rotation='horizontal', fontsize=5)
        axs[0].set_yticks(np.arange(len(indices)))
        axs[1].set_xticklabels(list(parcels.keys()), rotation='vertical', fontsize=5)
        axs[1].set_xticks(np.arange(len(indices)))
    plt.colorbar(mappable=scalarMap, ax=axs[:], shrink=0.5)
    print(plt.gcf())
    plt.show()


def describe_cohort_networks(network_name, subj_idx_list_1, subj_idx_list_2, conn_data=None, prop_thr=None, subject_level=False):
    conn_data = tan.get_conn_data() if conn_data is None else conn_data
    matrix_1 = get_cohort_network_matrices(network_name, subj_idx_list_1, subject_level=subject_level, conn_data=conn_data, prop_thr=prop_thr)
    matrix_2 = get_cohort_network_matrices(network_name, subj_idx_list_2, subject_level=subject_level, conn_data=conn_data, prop_thr=prop_thr)
    t_test_results = scipy.stats.ttest_ind(matrix_1, matrix_2, axis=None, nan_policy='omit')
    print(f'Shapes: {matrix_1.shape=} | {matrix_2.shape=}')
    print(f'Means: {np.nanmean(matrix_1)=} | {np.nanmean(matrix_2)=}')
    print(f'StDev: {np.nanstd(matrix_1)=} | {np.nanstd(matrix_2)=}')
    print(f'{t_test_results=}')


def get_cohort_comparison_over_thresholds(network_name, group_indices, group_names=None, thr_range=None,
                                          thr_increment=None, conn_data=None, subject_level=False,
                                          plot=False, exclude_negatives=False):
    conn_data = tan.get_conn_data() if conn_data is None else conn_data
    thr_increment = 0.1 if thr_increment is None else thr_increment
    thr_range = np.arange(0., 1, thr_increment) if thr_range is None else thr_range
    group_names = ['1', '2'] if group_names is None else group_names
    comp_df = pd.DataFrame(columns=['threshold', 'group', 'connectivity'])
    df_idx = 0
    for value in thr_range:
        network_mask = get_proportional_threshold_mask(network_name, value, exclude_negatives=exclude_negatives)
        matrix_1 = get_cohort_network_matrices(network_name, group_indices[0], subject_level=subject_level,
                                               conn_data=conn_data, network_mask=network_mask)
        matrix_2 = get_cohort_network_matrices(network_name, group_indices[1], subject_level=subject_level,
                                               conn_data=conn_data, network_mask=network_mask)
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
        plot_cohort_comparison_over_thresholds(network_name, comp_df, group_names)
    return comp_df


def plot_cohort_comparison_over_thresholds(network_name, comparison_df, group_names):
    ''' Plot group differences in connectivity strength over a range of thresholds.

        Parameters
        ----------
        network_name    : str, don't include "network"
        comparison_df   : pandas.Dataframe, output from get_cohort_over_thresholds()
        group_names     : list, should include two str items with the group names

        STILL NEEDS ADDRESSED
        ---------------------
        1. add * markers for significance testing above the error bars
        - The significance testing at a given thresh can be done like this (cdf = comparison_df):
        g1 = cdf[cdf['group'] == group_names[0]][cdf['threshold']==thresh]['connectivity']
        g2 = cdf[cdf['group'] == group_names[1]][cdf['threshold']==thresh]['connectivity']
        ttest_ind(g1, g2)
        - So this needs to loop over each threshold, calculate the p value, and then place the
          asterix in the right position
    '''
    fig, ax = plt.subplots()
    sns.lineplot(data=comparison_df, x='threshold', y='connectivity', hue='group', marker='.',
                 ci=95, err_style='bars', alpha=0.8, err_kws={'capsize':5}, linestyle=':')
    plt.title(f'Group Differences in {network_name} Network')
    ax.set_xlabel('Proportional Threshold')
    ax.set_ylabel('Connectivity Strength')
    plt.show()


def get_subject_scores(measure):
    scores = {}
    scores_df = pd.DataFrame(columns=['index', 'subject', measure])

    for row in subj_data.index:
        if not np.isnan(float(subj_data[subj_data.index == row][measure])):
            scores_df.loc[len(scores_df)] = [row,
                                             str(subj_data[subj_data.index == row]['subject'].values[0]),
                                             float(subj_data[subj_data.index == row][measure])]
            # scores[row] = float(subj_data[subj_data.index == row][measure])
    return scores_df


def plot_score_by_network(measure, network, drop=[], conn_data=None, prop_thr=None, network_mask=None,
                          exclude_negatives=False, stats=False):
    conn_data = tan.get_conn_data() if conn_data is None else conn_data
    scores_df = get_subject_scores(measure)
    conn_values = []
    for idx in drop:
        scores_df = scores_df[scores_df['index'] != idx]
    for subj in scores_df['index']:
        m = get_network_matrix(network, subj, conn_data=conn_data, prop_thr=prop_thr,
                               network_mask=network_mask, exclude_negatives=exclude_negatives)
        m[np.triu_indices(m.shape[0], k=0)] = np.nan
        conn_values.append(np.nanmean(m))

    scores_df['connectivity'] = conn_values
    sns.scatterplot(data=scores_df, x='connectivity', y=measure)
    if stats is True:
        print(pearsonr(scores_df['connectivity'], scores_df[measure]))
    return scores_df
