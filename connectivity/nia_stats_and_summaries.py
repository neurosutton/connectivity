"""
Common analytic tools for non-imaging comparisons.


Author: Brianne Sutton, PhD
Date: Dec 2020
version: 0.1
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fmri_analysis_plotting as faplot
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn import metrics
from tqdm import tqdm
import random


def summarize_group_differences(df, group_cols, msrs, graph=False):
    """
    Compare group differences and produce a table with the
    corresponding statistical tests.

    Parameters
    ----------
    df : long format dataframe
        Dataframe with measures for each person in the study to be
        compared along with any grouping labels.
    group_cols : str
        Name of the column(s) in the df that is (are) the grouper(s).
    msrs: list
        Metrics that are to be statistically compared.
    graph: boolean
        Returns a plot of group differences, if True.

    Returns
    -------
    result : dataframe
        Summary table of group differences with p-values

    (optional) plot of group differences
    """

    msrs = [msrs] if not isinstance(msrs, list) else msrs
    msr_dict = {msr: ['mean', 'std', 'count'] for msr in msrs}
    group_cols = [group_cols] if not isinstance(msrs, list) else group_cols
    groups = list(set(df[group_col[0]])) # Assumes the first grouper is the main grouper
 
    if len(group_cols) > 1 or 'threshold' in df.columns:
        if len(group_cols) > 1:
            second_grouping = list(set(df[group_cols[1]]))
        else:
            groups_cols[1] = 'threshold'
            second_grouping = list(set(df['threshold']))

        for sg,toss in enumerate(second_grouping):
            grp1_ix = df.loc[(df[group_cols[0]] == groups[0]) & (df[group_cols[1]] == second_grouping[sg]),:].index
            grp2_ix = df.loc[(df[group_cols[0]] == groups[1]) & (df[group_cols[1]] == second_grouping[sg]),:].index 
            result = df.loc[df[group_cols[1]] == second_grouping[sg],:].groupby(group_cols[0]).agg(msr_dict).round(2).T.unstack()
            result = _helper_sgd(df, grp1_ix, grp2_ix, result, msr_dict)
    else:
        grp1_ix = df.loc[(df[group_cols[0]] == groups[0])].index
        grp2_ix = df.loc[(df[group_cols[0]] == groups[1])].index                       
        result = df.groupby(group_cols).agg(msr_dict).round(2).T.unstack()
        result = _helper_sgd(df, grp1_ix, grp2_ix, result, msr_dict)
    print(result)

    if graph:
        # TODO See if the logic for the graph holds for the threshold/multigrouper case
        keep = msrs + [group_col]
        tmp = df[keep]
        sns.pairplot(tmp, hue=group_col, palette='winter')
        plt.xticks(rotation=80)
        plt.show()
    return result


def _helper_sgd(df, grp1_ix, grp2_ix, result, msr_dict):
    """
    Helper function for summarize_grou_differences.

    Inputs built to depend on arguments defined in 
    summarize_group_differences.
    Returns summarized df
    """
    for msr in msr_dict.keys():
        grp1 = df.loc[grp1_ix, msr].dropna()
        grp2 = df.loc[grp2_ix, msr].dropna()
        result.loc[msr, ('stats', 'pvalue')] = ttest_ind(
            grp1, grp2)[-1].round(3)
    return result


def calculate_auc(
        df,
        network=None,
        grouping_col='group',
        name_id_col='subject',
        bootstrap=5000,
        msrs=None,
        subgroups=None,
        exclude=None):
    """Perform permutation-based statistical testing of graph measure AUCs.

    Parameters
    ----------
    df : pandas.DataFrame
        Long format dataframe with subject identifiers and various
        measures that are to be permuted. In general, using this package will
        denote the permuted measures with gm for graph measure.
    network : string
    grouping_col : string
    name_id_col : string
    bootstrap : int
    msrs : list, optional
    subgroups :
    exclude : list of subject indices, optional

    Returns
    -------
    Nothing. Performs statistical tests and displays the output.
    # TO DO: Should this return the results in some format?
    """
    # In the whole brain case, method will fail without specifying dtype.
    df['network'] = df['network'].astype(str)
    if exclude is not None:
        try:
            df = df.query('subj_ix not in @exclude')
        except TypeError:
            print('No exclusions applied. Might not have passed ',
                  f'the right data type ({type(exclude)}).')
    if network:
        tmp = df[df['network'].str.contains(
            network, case=False)].dropna(how='all')
    else:
        tmp = df[df['network'].replace({'nan': np.NaN}).isna()]
    print(f'Drawing from {df.shape} data points')

    if not msrs:
        excl_cols = ['subj', 'index', grouping_col, 'group', 'threshold']
        msrs = [msr for msr in tmp.columns if (
            not any(substring in msr for substring in excl_cols))
                and (tmp[msr].dtype in [np.float64, np.int64])
                and (len(set(tmp[msr])) > 3)
                and (len(tmp[msr].dropna()) > 0)]
    print(f'Working through {sorted(msrs)} for FCN: {network}')

    for msr in sorted(msrs):

        if subgroups:
            # Needed for the case of splitting treatment or diagnosis group
            # into categories; two populations will have three or more labels,
            # so this will choose the two to be compared.
            tmp = tmp.loc[tmp[grouping_col].isin(subgroups), :]

        groups = list(set(tmp[grouping_col]))
        # List of subject IDs
        group1_members = set(
            tmp.loc[tmp[grouping_col] == groups[0], name_id_col])
        study_exp_auc_diff = auc_group_diff(
            tmp, group1_members, msr, group_match_col=name_id_col)

        if study_exp_auc_diff and (not np.isnan(study_exp_auc_diff)):
            print(f'{msr.upper()}')
            print(f'{groups[0]} - {groups[1]} = {study_exp_auc_diff}')
            permuted_diffs = []
            for c in tqdm(
                    range(
                        1,
                        bootstrap),
                    bar_format=('{desc:<5.5}{percentage:3.0f}%|',
                                '{bar:10}{r_bar}')):
                permuted_group1_members = random.sample(
                    set(tmp[name_id_col]), len(group1_members))
                permuted_diffs.append(
                    auc_group_diff(
                        tmp,
                        permuted_group1_members,
                        msr,
                        group_match_col=name_id_col))

            prms_lssr = len(
                [val for val in permuted_diffs if val < study_exp_auc_diff])
            try:
                print('The experimental AUC difference, ',
                      f'{study_exp_auc_diff.round(3)}, occurs ',
                      f'{round(prms_lssr/bootstrap*100,3)}% of the time in ',
                      'the boostrapped results.')
            except BaseException:
                print(f'The AUC difference, {study_exp_auc_diff.round(3)}, ',
                      ' beyond any bootstrapped result')
            faplot.plot_auc(
                study_exp_auc_diff,
                permuted_diffs,
                msr,
                network=network)


def auc_group_diff(df, group1_list, msr, group_match_col='subj'):
    group1_means = df.loc[df[group_match_col].isin(
        group1_list), ['threshold', msr]].groupby('threshold').mean().values
    group2_means = df.loc[~df[group_match_col].isin(
        group1_list), ['threshold', msr]].groupby('threshold').mean().values
    thrs = sorted(set(df['threshold'].dropna()))

    if len(thrs) > 2:
        group1_auc = metrics.auc(thrs, group1_means)
        group2_auc = metrics.auc(thrs, group2_means)
        return group1_auc - group2_auc
    else:
        print(f'{msr} did not have enough threshold data points ',
              'for this comparison.')
        print(df.loc[df[group_match_col].isin(group1_list), [
              'threshold', msr]].groupby('threshold').mean().values)
