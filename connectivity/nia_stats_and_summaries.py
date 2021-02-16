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


def summarize_group_differences(df, group_col, msrs, graph=False):
    msr_dict = {msr: ['mean', 'std', 'count'] for msr in msrs}
    result = df.groupby(group_col).agg(msr_dict).round(2).T.unstack()
    groups = list(set(df[group_col]))
    for msr in msr_dict.keys():
        grp1 = df.loc[(df[group_col] == groups[0]), msr].dropna()
        grp2 = df.loc[(df[group_col] == groups[1]), msr].dropna()
        result.loc[msr, ('stats', 'pvalue')] = ttest_ind(
            grp1, grp2)[-1].round(3)
    print(result)

    if graph:
        keep = msrs + [group_col]
        tmp = df[keep]
        sns.pairplot(tmp, hue=group_col, palette='winter')
        plt.xticks(rotation=80)
        plt.show()
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
    """ Input: Long format dataframe with subject identifiers and various
        measures that are to be permuted. In general, using this package will
        denote the permuted measures with gm for graph measure. """
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
                    bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
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
                print(
                    f"The experimental AUC difference, {study_exp_auc_diff.round(3)}, occurs {round(prms_lssr/bootstrap*100,3)}% of the time in the boostrapped results.")
            except BaseException:
                print(f'AUC difference beyond any bootstrapped result')
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
        print(f'{msr} did not have enough threshold data points for this comparison.')
        print(df.loc[df[group_match_col].isin(group1_list), [
              'threshold', msr]].groupby('threshold').mean().values)
