"""
Load data from CONN output and create shared variables for other modules in package.

Date: December 2020
@author: Brianne Sutton, PhD from Josh Bear, MD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from matplotlib import cm
import seaborn as sns


import fmri_analysis_load_funcs as faload
import fmri_analysis_get_data as get
import shared


def plot_score_by_network(
        measure,
        network,
        drop=[],
        conn_data=None,
        prop_thr=None,
        network_mask=None,
        exclude_negatives=False,
        stats=False):
    conn_data = get.get_conn_data() if conn_data is None else conn_data
    scores_df = get.get_subject_scores(measure)
    conn_values = []
    for idx in drop:
        scores_df = scores_df[scores_df['index'] != idx]
    for subj in scores_df['index']:
        m = get.get_network_matrix(
            network,
            subj,
            conn_data=conn_data,
            prop_thr=prop_thr,
            network_mask=network_mask,
            exclude_negatives=exclude_negatives)
        m[np.triu_indices(m.shape[0], k=0)] = np.nan
        conn_values.append(np.nanmean(m))

    scores_df['connectivity'] = conn_values
    sns.scatterplot(data=scores_df, x='connectivity', y=measure)
    if stats is True:
        print(stats.pearsonr(scores_df['connectivity'], scores_df[measure]))
    return scores_df


def plot_cohort_comparison_over_thresholds(
        comparison_df,
        network_name=None,
        group='group',
        y='connectivity'):
    ''' Plot group differences in connectivity strength over a threshold range.

        Parameters
        ----------
        network_name : str, don't include "network"
        comparison_df : pandas.Dataframe, output from get_cohort_over_thresholds()
        group_names : list, should include two str items with the group names
        y : measure from graph to plot

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
    comparison_df.loc[comparison_df.network == 'nan', 'network'] = 'whole_brain'
    network_name = 'whole_brain' if network_name is None else network_name
    # df = df.query('subj_ix not in @exclude')
    df = comparison_df.query('network == @network_name')
    sns.lineplot(
        data=df,
        x='threshold',
        y=y,
        hue=group,
        marker='.',
        ci=95,
        err_style='bars',
        alpha=0.8,
        err_kws={
            'capsize': 5},
        linestyle=':')
    plt.title(f'Group Differences in {network_name} Network')
    ax.set_xlabel('Proportional Threshold')
    ax.set_ylabel(y)
    plt.show()


def plot_network_matrix(network_name, subj_idx, conn_data=None):
    if not conn_data:
        conn_data = shared.conn_data
    parcels = get.get_network_parcels(conn_data, network_name, subj_idx)
    indices = list(parcels.values())
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(conn_data['Z'][:, :, subj_idx][np.ix_(indices, indices)])
    fig.colorbar(im)
    # plt.matshow(conn_data['Z'][:, :, 0][np.ix_(indices, indices)])
    # Let's adjust the tick labels.
    plt.title(f'Subject: {subj_idx} | Network: {network_name}')
    plt.xticks(
        np.arange(
            len(indices)), list(
            parcels.keys()), rotation='vertical')
    plt.yticks(
        np.arange(
            len(indices)), list(
            parcels.keys()), rotation='horizontal')
    # plt.colorbar()
    ax.tick_params(
        axis="x",
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=False)
    plt.show()


def plot_cohort_network_matrix(conn_data, network_name, subj_idx_list):
    cohort_matrices = []  # need to collect all the matrices to add
    for subj in subj_idx_list:
        cohort_matrices.append(
            get.get_network_matrix(
                mdata, network_name, subj))
    cohort_matrices = np.asarray(cohort_matrices)
    mean_matrix = np.nanmean(cohort_matrices, axis=0)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(mean_matrix)
    fig.colorbar(im)
    plt.show()


def plot_cohort_comparison_matrices(
        network_name,
        subj_idx_list_1,
        subj_idx_list_2,
        prop_thr=None,
        vmin=None,
        vmax=None,
        conn_data=None,
        mdata=None):
    conn_data = get.get_conn_data() if conn_data is None else conn_data
    mean_matrix_1 = get.get_cohort_network_matrices(
        network_name,
        subj_idx_list_1,
        mean=True,
        conn_data=conn_data,
        prop_thr=prop_thr)
    mean_matrix_2 = get.get_cohort_network_matrices(
        network_name,
        subj_idx_list_2,
        mean=True,
        conn_data=conn_data,
        prop_thr=prop_thr)
    vmin = np.min([np.nanmin(mean_matrix_1), np.nanmin(
        mean_matrix_2)]) if vmin is None else vmin
    vmax = np.max([np.nanmax(mean_matrix_1), np.nanmax(
        mean_matrix_2)]) if vmax is None else vmax
    boundary = np.max([np.absolute(vmin), np.absolute(vmax)])
    parcels = get.get_network_parcels(
        network_name, subj_idx_list_1[0], mdata=mdata)
    indices = list(parcels.values())
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=180)
    cmap = plt.get_cmap('Spectral')
    cNorm = colors.Normalize(vmin=-boundary, vmax=boundary)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    im1 = axs[0].matshow(mean_matrix_1, cmap=cmap, norm=cNorm)
    im2 = axs[1].matshow(mean_matrix_2, cmap=cmap, norm=cNorm)
    if len(mean_matrix_1[0]) < 25:
        axs[0].set_xticklabels(
            list(
                parcels.keys()),
            rotation='vertical',
            fontsize=5)
        axs[0].set_xticks(np.arange(len(indices)))
        axs[0].set_yticklabels(
            list(
                parcels.keys()),
            rotation='horizontal',
            fontsize=5)
        axs[0].set_yticks(np.arange(len(indices)))
        axs[1].set_xticklabels(
            list(
                parcels.keys()),
            rotation='vertical',
            fontsize=5)
        axs[1].set_xticks(np.arange(len(indices)))
    plt.colorbar(mappable=scalarMap, ax=axs[:], shrink=0.5)
    print(plt.gcf())
    plt.show()


def plot_auc(study_exp_auc_diff, permuted_diffs, msr, network=None):
    """Intended as a helper function for nia_stats_and_summaries (calculate_auc).
    Plots calculated study-related experimental differences and random,
    permuted differences for a given metric.
    """
    network = network if network else 'Whole_brain'
    fig, ax = plt.subplots()
    sns.histplot(x=permuted_diffs, kde=True)
    plt.axvline(study_exp_auc_diff, color='r', linewidth=5)
    plt.title(f'{network}:{msr}')
    plt.show()
