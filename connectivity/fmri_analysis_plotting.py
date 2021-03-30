"""
Load data from CONN output and create shared variables for other modules in package.

Date: December 2020
@author: Brianne Sutton, PhD from Josh Bear, MD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import colors
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from matplotlib import cm
import seaborn as sns
from collections import OrderedDict


import fmri_analysis_load_funcs as faload
import fmri_analysis_get_data as get
import nia_stats_and_summaries as nss
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
        network=None,
        group='group',
        y='connectivity',
        exclude=None,
        threshold_range=None,
        normalize=False):
    ''' Plot group differences in connectivity strength over a threshold range.

        Parameters
        ----------
        network_name : str, don't include "network"
        comparison_df : pandas.Dataframe, output from get_cohort_over_thresholds()
        group_names : list, should include two str items with the group names
        y : measure from graph to plot
    '''

    if exclude is not None:
        try:
            comparison_df = comparison_df.query('subj_ix not in @exclude')
            print(set(comparison_df['subj_ix']))
        except TypeError:
            print('No exclusions applied. Might not have passed ',
                  f'the right data type ({type(exclude)}).')
    if threshold_range is not None:
        comparison_df = comparison_df.query(
            'threshold >= @threshold_range[0] and threshold <= @threshold_range[1]')

    fig, ax = plt.subplots()
    comparison_df.loc[comparison_df.network == 'nan',
                      'network'] = 'whole_brain'
    network = 'whole_brain' if network is None else network
    df = comparison_df.query('network == @network')
    if normalize:
        df = nss.normalize(df)
        y = y + '_normed'
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
    star_dict = add_asterisks(df, y, group=group)
    for thr in star_dict.keys():
        ax.text((thr - .01), star_dict[thr]['y'], '*')
    #ax.set_yscale('log', base=10) #
    plt.title(f'Group Differences in {network} Network')
    ax.set_xlabel('Density')
    ax.set_ylabel(y)
    plt.show()


def add_asterisks(df, msr, group='group'):
    """
    Add significance markers at each threshold on a graph.
    Parameters
    ----------
    df : DataFrame
        Pared down dataframe that contains the network of interest at any thresholds that
        are calculated. If the df has been normalized, then those values should be used.
    msr : string
        Name of the column that contains the data that will be plotted on the y-axis
    Returns
    -------

    """
    star_dict = {}
    group_names = list(set(df[group]))
    for thr in sorted(set(df['threshold'])):
        g1 = df.loc[(df[group] == group_names[0]) &
                    (df['threshold'] == thr), msr]
        g2 = df.loc[(df[group] == group_names[1]) &
                    (df['threshold'] == thr), msr]
        x, pval = scipy.stats.ttest_ind(g1, g2)
        if pval <= .05:
            star_dict[thr] = {}
            top_val = max(np.mean(g1), np.mean(g2))
            sd = max(np.std(g1), np.std(g2))
            star_dict[thr]['y'] = (top_val + sd) + (.03 * (top_val + sd))
            star_dict[thr]['pval'] = pval
    return star_dict


def plot_network_matrix(
        matrix=None,
        network_name=None,
        subj_idx=None,
        conn_data=None,
        id_networks=False,
        clear_triu=True,
        vmin=None,
        vmax=None,
        cmap=None):
    """

    """
    network_name = 'whole_brain' if not network_name else network_name
    conn_data = get.get_conn_data(
        clear_triu=clear_triu) if not conn_data else conn_data
    parcels = get.get_network_parcels(network_name)
    # Organize the array by FCN/alphabetical regions for grouped visualization
    # TODO check that the following line actually sorts properly
    # Noticing that single subject matrices are rearranged if it is uncommented
    #parcels = sorted(parcels.items())
    # Don't use ".items()", if the items are already pulled in the line above.
    indices = np.array([y for (x, y) in parcels.items()])
    vmin = -1 if vmin is None else vmin
    vmax = 1 if vmin is None else vmax
    cmap = plt.cm.seismic if cmap is None else cmap

    # matrix = conn_data[:, :, subj_idx][np.ix_(
    #     indices, indices)] if matrix is None else matrix

    if matrix is None:
        if subj_idx is None:
            raise ValueError(
                'Unsuccessful. Need to pass either a matrix or subj_idx.')
            return
        else:
            matrix = conn_data[:, :, subj_idx]
    else:
        matrix = np.nansum([matrix, matrix.T], axis=0) - \
            np.diag(np.diag(matrix))

    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(matrix[np.ix_(indices, indices)], cmap=cmap)
    im.set_clim(vmin, vmax)
    fig.colorbar(im)
    # Let's adjust the tick labels.
    plt.title(f'Subject: {subj_idx} | Network: {network_name}')
    '''
    if len(indices) < 20:
        plt.xticks(
            np.arange(
                len(indices)), list(
                parcels.keys()), rotation='vertical')
        plt.yticks(
            np.arange(
                len(indices)), list(
                parcels.keys()), rotation='horizontal')
    else:
        # Sanity check for what is represented in the matrix.
        # Can be deleted in the future.
        check_parc_order = []
        for i in indices:
            check_parc_order.append([k for k, v in parcels if v == i])
        print(check_parc_order)
    '''
    ax.tick_params(
        axis="x",
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=False)
    if network_name in ['wb', 'whole_brain', 'whole brain'] and id_networks:
        codes, vertices = add_squares()
        path = Path(vertices, codes)
        pathpatch = PathPatch(path, facecolor='None', edgecolor='red')
        ax.add_patch(pathpatch)
    plt.show()
    # print(
    # f'Order for FCN along diagonal is:\n{sorted(set([fcn.split("_")[0] for
    # fcn in parcels.keys()]))}')


def add_squares(network='wb'):
    """
    Plot squares around network nodes using naming convention and strategy from
    https://matplotlib.org/stable/gallery/shapes_and_collections/compound_path.html#sphx-glr-gallery-shapes-and-collections-compound-path-py

    Parameters
    ----------
    None

    Returns
    -------
    vertices and codes to use with matplotlib.path to draw squares
    around networks.
    """

    parcels = get.get_network_parcels(network)
    # Organize the array by FCN/alphabetical regions for grouped visualization
    parcels = OrderedDict(sorted(parcels.items()))
    unique_networks = set([k.split("_")[0] for k in parcels.keys()])
    # Turn the parcel OrderedDict into a df, so that network corners
    # can be identified by value in index (not np_ix, which is where
    # the data is pulled from in mdata)
    df = pd.DataFrame(parcels.items(), columns=['label', 'np_ix'])
    codes = []
    vertices = []
    for n in unique_networks:
        ix = df.loc[df['label'].str.contains(n)].index
        corner1 = (np.min(ix), np.min(ix))
        corner2 = (np.min(ix), np.max(ix))
        corner3 = (np.max(ix), np.max(ix))
        corner4 = (np.max(ix), np.min(ix))

        codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        vertices += (corner1, corner2, corner3, corner4, corner1)
    return codes, vertices


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
    network = network if network else 'whole_brain'
    fig, ax = plt.subplots()
    sns.kdeplot(
        x=permuted_diffs,
        fill=True,
        linewidth=0,
        alpha=.8,
        color=sns.xkcd_rgb['light grey'])
    vert_lines = find_stdevs(permuted_diffs)
    for val in vert_lines:
        plt.axvline(val, ymin=0, ymax=1, color='w', linewidth=2, ls='--')
    plt.axvline(
        np.mean(permuted_diffs),
        ymin=0,
        ymax=1,
        color='w',
        linewidth=2,
        ls='-')
    plt.axvline(
        study_exp_auc_diff,
        ymin=0,
        ymax=0.625,
        color='b',
        linewidth=2,
        ls='-')
    label = f'Experimental value\n{round(study_exp_auc_diff,3)}'
    ymin, ymax = ax.get_ylim()
    ax.annotate(label,
                xy=(study_exp_auc_diff, .65 * ymax),
                xytext=(study_exp_auc_diff, .78 * ymax),
                arrowprops=dict(width=2,
                                headwidth=8,
                                facecolor='blue',
                                shrink=.05),
                horizontalalignment='center')
    plt.title(f'{network}:{msr}')
    sns.despine()
    plt.show()


def find_stdevs(sample):
    mean = np.mean(sample)
    sd = np.std(sample)
    return (
        mean - 3 * sd,
        mean - 2 * sd,
        mean - sd,
        mean + sd,
        mean + 2 * sd,
        mean + 3 * sd)
