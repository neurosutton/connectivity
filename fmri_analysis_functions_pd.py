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
import seaborn as sns
from matplotlib import cm
import json

with open(op.join(op.dirname(op.realpath(__file__)),'directory_defs.json')) as f:
    defs = json.load(f)
    conn_dir = defs['conn_dir']
    main_dir = defs['main_dir']
data_dir = op.join(conn_dir)
conn_file = 'resultsROI_Condition001.mat'
subjects_file =  op.join(main_dir,'eses_subjects_202008.csv')

name_id_col = "BK_name"
group_id_col = "group"
debug = ''

def load_mat(data_dir, conn_file):
    """Loading and reloading the module is much quicker with loading the matrix as its own method. Call first, so that there is data, though."""
    mdata = loadmat(op.join(data_dir, conn_file))
    rois = mdata['names']
    rois = [roi[0].replace('hcp_atlas.','') for r_array in rois for roi in r_array]
    return mdata, rois

def get_subj_df_data(subjects_file = subjects_file):
    """Primarily for reading in demographic and neuropsychological data."""
    subj_df = pd.DataFrame(pd.read_csv(subjects_file))
    subj_dict = {k:v for k,v in enumerate(subj_df[name_id_col])}
    group_dict = dict(zip(subj_df[name_id_col], subj_df[group_id_col]))
    return subj_df, subj_dict, group_dict

def get_group_membership(subj_df, group):
    if debug:
        print(subj_df.loc[subj_df[group_id_col]==group,group_id_col])
    else:
        pass
    return subj_df.index[subj_df[group_id_col]==group].tolist()


def get_group_membership_by_name(subj_df, group):
    return subj_df[name_id_col].loc[:,subj_df[group_id_col] == group]

def get_network_parcels(mdata, network_name):
    """Returns parcel names and indices with HCP remaining in the name and indexed to work with numpy-based functions."""
    parcel_names = [str[0] for str in mdata['names'][0]]
    parcels = {k:v for v,k in enumerate([str[0] for str in mdata['names'][0]])}
    pattern = 'hcp_atlas.' + network_name + '*'
    matching = fnmatch.filter(parcels.keys(), pattern)
    network_parcels = {k:v for k,v in parcels.items() if k in matching}
    indices = [parcels.get(key) for key in matching]
    return network_parcels


def get_parcel_dict(mdata, network_name=None, inverse=False):
    """Alternate method to get ROI indices and names."""
    parcel_names = [str[0] for str in mdata['names'][0]]
    parcel_dict = {}
    parcel_dict_inv = {}
    if network_name:
        print(f'Selecting ROIs belonging to the {network_name} network.\n')
    for p,parcel in enumerate(parcel_names):
        parcel = parcel.replace('hcp_atlas.','') # Clean the names
        if network_name:
            if network_name.lower() in parcel.lower():
                parcel_dict[parcel] = p
            elif debug:
                print(f'Did not find {network_name} in {parcel}')
            else:
                pass
        else:            
            parcel_dict[parcel] = p
    if inverse == True:
        parcel_dict = {v:k for k,v in parcel_dict.items()}
    if debug:
        print(f'Search "{network_name}" returned these ROIs:\n{parcel_dict}')
    return parcel_dict

def create_conn_df(mdata, triu=False):
    """"Create the full, filterable connectivity matrix. Add subject id and group info"""
    subj_ix = mdata['Z'].shape[-1]
    x, subj_dict, group_dict = get_subj_df_data(subjects_file)
    parcel_dict = get_parcel_dict(mdata, network_name=None)
    rois = list(parcel_dict.keys())
    col_names = [name_id_col] + rois # Use the subj column to be able to search and filter by specific participants
    conn_df = pd.DataFrame(columns = col_names)
    for s in range(0, subj_ix):
        tmp_df = pd.DataFrame(mdata['Z'][:,:mdata['Z'].shape[0],s], index = rois, columns = rois) # Grab the ROIs that are part of the atlas and not all the extra regressor correlations
        if triu==True:
            tmp = tmp_df.to_numpy()
            tmp[np.triu_indices(tmp.shape[0], k=0)] = np.nan
            tmp_df = pd.DataFrame(data=tmp, index = rois, columns=rois)
        tmp_df[name_id_col] = subj_dict[s]
        conn_df = pd.concat([conn_df,tmp_df])

    conn_df[group_id_col] = conn_df[name_id_col].map(group_dict)
    return conn_df
    

def get_network_matrix(mdata, network_name=None, subj_list=None, parcel_dict=None, triu=False):
    """Provides the overarching connectivity for all participants as a searchable dataframe. No group assignments or covariates are included by this method."""
    if not parcel_dict:
        parcel_dict = get_parcel_dict(mdata, network_name)
    # Select the index for the third dimension fo the numpy array
    if subj_list:
        if not isinstance(subj_list,list):
            subj_list = [subj_list]

    # Apply filters to the connectivity dataframe
    conn_df = create_conn_df(mdata,triu)
    if subj_list:
        print(f'Gathering {subj_list}')
    if network_name:
        cols = [col for col in conn_df.columns if col in ([name_id_col, group_id_col] + list(parcel_dict.keys()))]
        conn_df = conn_df[cols][conn_df.index.isin(parcel_dict.keys())]
        print(f'After applying {network_name} filter, the matrix is {conn_df.shape}')
    if subj_list:
        if isinstance(subj_list[0], int):
            conn_df = conn_df.iloc[subj_list,:]
        else:
            conn_df = conn_df.loc[conn_df[name_id_col].isin(subj_list),:]
        if debug:
            print(subj_list)
        print(f'After filtering for {len(subj_list)} participants, the matrix is {conn_df.shape}\n')
    return conn_df


def get_cohort_network_matrices(mdata, network_name, group, mean=False, triu=False):
    ''' Get the matrices for a cohort of patients in a given network. '''
    cohort_df = get_network_matrix(mdata,network_name)
    cohort_df = cohort_df.loc[cohort_df[group_id_col]==group,:]
    cohort_df.drop(columns = [name_id_col,group_id_col], inplace = True)
    print(f'After group filter, matrix size is {cohort_df.shape}')
    if mean is True:
        if debug:
            print(cohort_df.groupby(level=0).mean())
        return cohort_df.groupby(level=0).mean().to_numpy()
        #return np.nanmean(cohort_df.to_numpy(), axis=0) # This flattened the array and didn't seem to return the means by ROI
    else:
        return cohort_df.to_numpy()


def check_subj_avlbl(subj):
    """"Figure out whether the given subject input is valid for the study."""
    x, subj_dict, y = get_subj_df_data()
    if subj not in subj_dict.values():
        return_status = 0
        print(f'Subject {subj} is not present in the sample.\nChoose from {subj_dict.values()} and re-run')
    else:
        return_status = 1
    return return_status

def plot_network_matrix(mdata, network_name, subj, parcel_dict=None):
    if subj:
        status = check_subj_avlbl(subj)
    else:
        status = 1
    if status == 1:
        if not parcel_dict:
            parcels = get_parcel_dict(mdata, network_name)
        fig = plt.figure()
        ax = plt.gca()
        df = get_network_matrix(mdata, network_name, subj)
        if isinstance(df, pd.DataFrame):
            df.drop(columns = [name_id_col,group_id_col], inplace = True)
            df = df.to_numpy()
        im = ax.matshow(df)
        fig.colorbar(im)
        # Let's adjust the tick labels.
        plt.title(f'Subject: {subj} | Network: {network_name}')
        plt.xticks(np.arange(len(parcels.keys())), list(parcels.keys()), rotation='vertical')
        plt.yticks(np.arange(len(parcels.keys())), list(parcels.keys()), rotation='horizontal')
        # plt.colorbar()
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
        plt.show()


def plot_cohort_network_matrix(mdata, network_name, group):
    mean_matrix = get_cohort_network_matrices(mdata, network_name, group, mean = True)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(mean_matrix)
    fig.colorbar(im)
    parcels = get_parcel_dict(mdata, network_name)
    plt.xticks(np.arange(len(parcels.keys())), list(parcels.keys()), rotation='vertical')
    plt.yticks(np.arange(len(parcels.keys())), list(parcels.keys()), rotation='horizontal')
    plt.show()

def plot_cohort_comparison(mdata, network_name, group_1, group_2, vmin=None, vmax=None):
    mean_matrix_1 = get_cohort_network_matrices(mdata, network_name, group_1, mean=True)  # need to collect all the matrices to add
    mean_matrix_2 = get_cohort_network_matrices(mdata, network_name, group_2, mean=True)
    vmin = np.min([np.nanmin(mean_matrix_1), np.nanmin(mean_matrix_2)]) if vmin is None else vmin
    vmax = np.max([np.nanmax(mean_matrix_1), np.nanmax(mean_matrix_2)]) if vmax is None else vmax
    boundary = np.max([np.absolute(vmin), np.absolute(vmax)])
    inv_parcels = get_parcel_dict(mdata, network_name, inverse = True)
    indices = list(inv_parcels.values())
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=180)
    cmap = plt.get_cmap('Spectral')
    cNorm = colors.Normalize(vmin=-boundary, vmax=boundary)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    im1 = axs[0].matshow(mean_matrix_1, cmap=cmap, norm=cNorm)
    im2 = axs[1].matshow(mean_matrix_2, cmap=cmap, norm=cNorm)
    if len(mean_matrix_1[0]) < 25:
        axs[0].set_xticklabels(list(inv_parcels.keys()), rotation='vertical', fontsize=5)
        axs[0].set_xticks(np.arange(len(indices)))
        axs[0].set_yticklabels(list(inv_parcels.keys()), rotation='horizontal', fontsize=5)
        axs[0].set_yticks(np.arange(len(indices)))
        axs[1].set_xticklabels(list(inv_parcels.keys()), rotation='vertical', fontsize=5)
        axs[1].set_xticks(np.arange(len(indices)))
    plt.colorbar(mappable=scalarMap, ax=axs[:], shrink=0.5)
    print(plt.gcf())
    plt.show()


def describe_cohort_networks(mdata, network_name, group_1, group_2, name_1=None, name_2=None):
    matrix_1 = get_cohort_network_matrices(mdata, network_name, group_1, mean=False, triu = True)
    matrix_2 = get_cohort_network_matrices(mdata, network_name, group_2, mean=False, triu = True)
    if debug:
        print(matrix_1, matrix_2)
    # Need to mask out the upper triangle of all of these.
    t_test_results = scipy.stats.ttest_ind(matrix_1, matrix_2, axis=None, nan_policy='omit')
    print(f'Shapes: {matrix_1.shape} | {matrix_2.shape}')
    print(f'Means: {np.nanmean(matrix_1)} | {np.nanmean(matrix_2)}')
    print(f'StDev: {np.nanstd(matrix_1)} | {np.nanstd(matrix_2)}')
    print(f'{t_test_results}')

def plot_score_by_network(subjects_file, measure, mdata, network, drop=[]):
    score_df = get_subj_df_data(subjects_file)
    col = [col for col in score_df.columns if measure in col]
    if len(col) > 1:
        print(f'Found multiple matching columns for {measure}')
    scores = dict(zip(score_df[name_id_col],score_df[col]))
    for idx in drop:
        scores.pop(idx, None)
    for subj in scores.keys():
        m = get_network_matrix(mdata, network, subj)
        m[np.triu_indices(m.shape[0], k=0)] = np.nan
        plt.scatter(np.nanmean(m), scores[subj])
    plt.show()

def import_conn_mat(in_file, mstr_df, mdata):
    # Load in the data, initially including the extra correlations
    df = pd.DataFrame( data = mdata['Z'],
                      index = roi_labels)

    # Keep only the HCP ROIs
    df = df.iloc[:,:df.shape[0]] # pare down the values to the ROI-to-ROI of interest, not all other covariates and/or atlases

    # Undo the z transformation - FOR COMPARISON WITH JOSH, COMMENT THIS OUT
    #df.applymap(lambda x: np.tanh(x))  # Get r values instead of z-scores

    # Add labels for the HCP ROIs. Per Josh's observation, change the logic to grab the names from names2, rather than assume that the ROIs are the same going down and across
    rois_col = mdata['names'] # Retrieve the ROI names for the HCP atlas
    roi_labels = [roi[0].replace('hcp_atlas.','') for r_array in rois_col for roi in r_array]
    rois_col = mdata['names2'] # Retrieve the ROI names for the HCP atlas
    roi_labels_col = [roi[0].replace('hcp_atlas.','') for r_array in rois_col for roi in r_array]
    df.columns = roi_labels_col[:df.shape[0]]

    plt.matshow(df) # Sanity check that the tanh(x) function returns a symmetrical, logical matrix.
    plt.colorbar()
    plt.show()

    # Add the subject number to the long df prior to concatenation
    tmp = op.basename(in_file)
    df[name_id_col] = tmp.split('_')[1]
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
