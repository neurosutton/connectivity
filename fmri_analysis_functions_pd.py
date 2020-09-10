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

with open(op.join(op.realpath(__file__),'directory_defs.json') as f:
    defs = json.load(f)
    conn_dir = defs['conn_dir']

data_dir = op.join(conn_dir,'conn_project01_results')
file = 'resultsROI_Condition001.mat'
subject_file = op.join(conn_dir,'eses_subjects_202008.csv')

mdata = loadmat(op.join(data_dir, file))

# Numpy version
#subj_data = pd.read_csv(subject_file)
#eses_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'eses']
#hc_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'hc']

def get_subj_df_data(subject_file):
    """Primarily for reading in demographic and neuropsychological data."""
    subj_df = pd.DataFrame(pd.read_csv(subject_file))
    #dict(zip(subj_df['conn_name'],subj_df['BK_name']))
    subj_dict = {k,v for k,v in enumerate(subj_df['BK_name'])}
    return subj_df,

def get_group_membership(subj_df, group):
    return subj_df.loc[:,subj_df['group'] == group].ix

def get_network_parcels(conn_data, network_name, subj_idx):
    """Returns parcel names and indices with HCP remaining in the name and indexed to work with numpy-based functions."""
    parcel_names = [str[0] for str in conn_data['names'][0]]
    parcels = {k:v for v,k in enumerate([str[0] for str in conn_data['names'][0]])}
    pattern = 'hcp_atlas.' + network_name + '*'
    matching = fnmatch.filter(parcels.keys(), pattern)
    network_parcels = {k:v for k,v in parcels.items() if k in matching}
    indices = [parcels.get(key) for key in matching]
    return network_parcels


def get_parcel_dict(conn_data, network_name=None):
    """Alternate method to get ROI indices and names."""
    parcel_names = [str[0] for str in conn_data['names'][0]]
    parcel_dict = {}
    parcel_dict_inv = {}
    for p,parcel in enumerate(parcel_names):
        parcel = parcel.replace('hcp_atlas.','') # Clean the names
        if network_name in parcel: # Check that this works with the whole brain, i.e., no network
            parcel_dict[parcel] = p
            parcel_dict_inv[p] = parcel
    return parcel_dict, parcel_dict_inv


def create_conn_df(mdata):
    """Provides the overarching connectivity for all participants as a searchable dataframe"""
    x, subj_dict = get_subj_df_data(subject_file)
    parcel_dict = get_parcel_dict(conn_data)
    col_names = parcel_dict.keys()
    conn_df = pd.DataFrame(columns = col_names.append('subj'))
    for s in range(0,mdata['Z'].shape[-1]):
        tmp_df = pd.DataFrame(mdata['Z'][:,mdata['Z'].shape[0],s], index = col_names, columns = col_names)
        tmp_df['subj'] = subj_dict[s]
        conn_df = pd.concat(conn_df,tmp_df)



def get_network_matrix(conn_data, network_name, subj):
    parcels = get_parcel_dict(conn_data, network_name)
    indices = list(parcels.values())
    return conn_data['Z'][:, :, subj_idx][np.ix_(indices, indices)]


def get_cohort_network_matrices(conn_data, network_name, subj_idx, mean=False):
    ''' Get the matrices for a cohort of patients in a given network. '''
    cohort_matrices = []  # need to collect all the matrices to add
    for subj in subj_idx:
        cohort_matrices.append(get_network_matrix(mdata, network_name, subj))
    cohort_matrices = np.asarray(cohort_matrices)
    if mean is True:
        return np.nanmean(cohort_matrices, axis=0)
    else:
        return cohort_matrices


def plot_network_matrix(conn_data, network_name, subj_idx):
    parcels = get_parcel_dict(conn_data, network_name)
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


def plot_cohort_comparison(conn_data, network_name, subj_idx_list_1, subj_idx_list_2, vmin=None, vmax=None):
    mean_matrix_1 = get_cohort_network_matrices(conn_data, network_name, subj_idx_list_1, mean=True)  # need to collect all the matrices to add
    mean_matrix_2 = get_cohort_network_matrices(conn_data, network_name, subj_idx_list_2, mean=True)
    vmin = np.min([np.nanmin(mean_matrix_1), np.nanmin(mean_matrix_2)]) if vmin is None else vmin
    vmax = np.max([np.nanmax(mean_matrix_1), np.nanmax(mean_matrix_2)]) if vmax is None else vmax
    boundary = np.max([np.absolute(vmin), np.absolute(vmax)])
    parcels = get_network_parcels(conn_data, network_name, subj_idx_list_1[0])
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


def describe_cohort_networks(conn_data, network_name, subj_idx_list_1, subj_idx_list_2, name_1=None, name_2=None):
    matrix_1 = get_cohort_network_matrices(conn_data, network_name, subj_idx_list_1, mean=False)
    matrix_2 = get_cohort_network_matrices(conn_data, network_name, subj_idx_list_2, mean=False)
    # Need to mask out the upper triangle of all of these.
    for m in matrix_1:
        m[np.triu_indices(m.shape[0], k=0)] = np.nan
    t_test_results = scipy.stats.ttest_ind(matrix_1, matrix_2, axis=None, nan_policy='omit')
    print(f'Shapes: {matrix_1.shape=} | {matrix_2.shape=}')
    print(f'Means: {np.nanmean(matrix_1)=} | {np.nanmean(matrix_2)=}')
    print(f'StDev: {np.nanstd(matrix_1)=} | {np.nanstd(matrix_2)=}')
    print(f'{t_test_results=}')


def get_subject_scores(subject_file, measure):
    subj_data = pd.read_csv(subject_file)
    scores = {}
    for row in subj_data.index:
        if not np.isnan(float(subj_data[subj_data.index == row][measure])):
            scores[row] = float(subj_data[subj_data.index == row][measure])
    return scores


def plot_score_by_network(subject_file, measure, conn_data, network, drop=[]]):
    scores = get_subject_scores(subject_file, measure)
    for idx in drop:
        scores.pop(idx, None)
    for subj in scores.keys():
        m = get_network_matrix(conn_data, network, subj)
        m[np.triu_indices(m.shape[0], k=0)] = np.nan
        plt.scatter(np.nanmean(m), scores[subj])
    plt.show()

def import_conn_mat(in_file, mstr_df):
    # Load .mat file
    mdata = scipy.io.loadmat(in_file)
    rois = mdata['names'] # Retrieve the ROI names for the HCP atlas

    # Clean up the names a little
    roi_labels = [roi[0].replace('hcp_atlas.','') for r_array in rois for roi in r_array]
    print('Found {} ROIs'.format(len(roi_labels)))

    # Load in the data, initially including the extra correlations
    df = pd.DataFrame( data = mdata['Z'],
                      index = roi_labels)

    # Keep only the HCP ROIs
    df = df.iloc[:,:df.shape[0]] # pare down the values to the ROI-to-ROI of interest, not all other covariates and/or atlases

    # Undo the z transformation - FOR COMPARISON WITH JOSH, COMMENT THIS OUT
    #df.applymap(lambda x: np.tanh(x))  # Get r values instead of z-scores

    # Add labels for the HCP ROIs. Per Josh's observation, change the logic to grab the names from names2, rather than assume that the ROIs are the same going down and across
    rois_col = mdata['names2'] # Retrieve the ROI names for the HCP atlas
    roi_labels_col = [roi[0].replace('hcp_atlas.','') for r_array in rois_col for roi in r_array]
    df.columns = roi_labels_col[:df.shape[0]]

    plt.matshow(df) # Sanity check that the tanh(x) function returns a symmetrical, logical matrix.
    plt.colorbar()
    plt.show()

    # Add the subject number to the long df prior to concatenation
    tmp = os.path.basename(in_file)
    df['subj'] = tmp.split('_')[1]
    print('Adding {}'.format(tmp.split('_')[1]))
    mstr_df = pd.concat([mstr_df, df]) # stack the data on top of each other, so it is searchable by name
    return mstr_df

def plot_correl_matrix(corr,correl_type='beta'):

    # Mask diagonal
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True # Finds the indices of the upper triangle

    # Define plot properties
    if correl_type == 'F':
        vmin    = 0
        vmax    = 10
        annot   = False
        cmap    = "coolwarm"

    elif correl_type == 'beta' :
        vmin    = -0.5
        vmax    = 1
        annot   = False #True
        cmap    = "viridis" #"RdBu_r"

    elif correl_type == 'p' :
        vmin    = 0
        vmax    = 0.1
        annot   = False
        cmap    = "warmcool"

    f, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr, mask=mask, vmin=vmin,  vmax=vmax, square=True, cmap=cmap, annot=annot, cbar=True, xticklabels=True, yticklabels=True, linewidths=.0 )
    #sns.heatmap(corr, vmin=vmin,  vmax=vmax, square=True, cmap=cmap, annot=annot, cbar=True, xticklabels=True, yticklabels=True, linewidths=.0 )

    plt.xticks(rotation=80)
    plt.yticks(rotation=0)
    plt.ylabel('')
    plt.xlabel('')
    plt.show()
    #plt.clf()
