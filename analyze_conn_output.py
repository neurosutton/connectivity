import scipy.io
import os.path as op
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn

def get_connectivity_matrix(file):
    return scipy.io.loadmat(file)['Z']


def upper_tri_file(file):
    m = get_connectivity_matrix(file)
    m = np.triu(m)
    m[np.tril_indices(m.shape[0], 0)] = np.nan
    return m


def plot_connectivity_matrix(file):
    m = upper_tri_file(file)
    plt.matshow(m)
    plt.colorbar()
    plt.show()


def get_connectivity_mean(file):
    m = upper_tri_file(file)
    return np.nanmean(m)


def get_connectivity_std(file):
    m = upper_tri_file(file)
    return np.nanstd(m)


def get_roi_to_roi_value(file, roi1, roi2):
    m = get_connectivity_matrix(file)
    return m[roi1, roi2]


def get_roi_to_roi_values_for_list(file_list, roi1, roi2):
    values = np.empty(len(file_list))
    for idx, file in enumerate(file_list):
        values[idx] = get_roi_to_roi_value(file, roi1, roi2)
    return values


def get_submatrix_from_rois(file, rois):
    m = get_connectivity_matrix(file)
    m_sub = np.zeros((len(rois), len(rois)))
    for idx1, i1 in enumerate(rois):
        for idx2, i2 in enumerate(rois):
            m_sub[idx1, idx2] = m[i1, i2]
    m_sub = np.triu(m_sub)
    m_sub[np.tril_indices(m_sub.shape[0], 0)] = np.nan
    return m_sub


def get_network_values_for_list(file_list, rois):
    all_values = []
    for file in file_list:
        file_values = get_submatrix_from_rois(file, rois)
        all_values = all_values + list(file_values.flatten())
    return all_values


def get_network_mean(file_list, rois):
    return np.nanmean(get_network_values_for_list(file_list, rois))


def test_network_differences(files_list_1, files_list_2, rois):
    values_1 = get_network_values_for_list(files_list_1, rois)
    values_2 = get_network_values_for_list(files_list_2, rois)
    return stats.ttest_ind(values_1, values_2, nan_policy='omit')


def plot_network_differences(network_list_1, network_list_2, labels, ymax=None):
    list_1_means = []
    list_2_means = []

    for network in network_list_1:
        list_1_means.append(np.nanmean(network))
    for network in network_list_2:
        list_2_means.append(np.nanmean(network))
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, list_1_means, width, label='ESES')
    rects2 = ax.bar(x + width/2, list_2_means, width, label='HC')

    ax.set_ylabel('Intranetwork Strength')
    ax.set_title('Connectivity by Network and Group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    if ymax is not None:
        ax.set_ylim([0, ymax])
    fig.tight_layout()
    plt.show()


def plot_connectivity_values(files_list, rois):
    values = []
    for file in files_list:
        values.append(get_network_mean([file], rois))
    x = np.arange(len(files_list))
    fig, ax = plt.subplots()
    ax.bar(x, values, 0.35)
    plt.show()

def plot_connectivity_against_variable(files_list, rois, title, var_name, values, verbose=False):
    conn_values = []
    for file in files_list:
        conn_values.append(get_network_mean([file], rois))
    if verbose:
        print(stats.pearsonr(values, conn_values))
    fig, ax = plt.subplots()
    ax.scatter(values, conn_values)
    ax.set_xlabel(var_name)
    ax.set_ylabel('Network Strength')
    plt.title(title)
    plt.show()

def plot_connectivity_seaborn(files_list, rois, title, var_name, values, verbose=False, print_corr=False):
    conn_values = []
    for file in files_list:
        conn_values.append(get_network_mean([file], rois))
    corr, pvalue = stats.pearsonr(values, conn_values)
    if verbose:
        print(stats.pearsonr(values, conn_values))
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=227)
    seaborn.set_style('whitegrid')
    seaborn.regplot(values, conn_values, ci=None, ax=ax, marker='.',
                    truncate=False,
                    line_kws={'color':'k','alpha':0.7, 'lw':1},
                    scatter_kws={'s': 100})
    ax.set_xlabel(var_name)
    ax.set_ylabel('Network Strength')
    ax.text(30, 0.5, 'r=' + str(round(corr, 2)) + ', p-value=' + str(round(pvalue, 2)))
    plt.title(title)
    plt.show()
