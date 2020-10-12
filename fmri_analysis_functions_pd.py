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
import fnmatch, random, time
from matplotlib import colors
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
from matplotlib import cm
import json
import bct
from collections import OrderedDict, defaultdict
from datetime import datetime
from sklearn import preprocessing, metrics, model_selection
from tqdm import tqdm


with open(op.join(op.dirname(op.realpath(__file__)),'directory_defs.json')) as f:
    defs = json.load(f)
    conn_dir = defs['conn_dir']
    main_dir = defs['main_dir']
data_dir = op.join(conn_dir)
conn_file = 'resultsROI_Condition001.mat'
subjects_file =  op.join(main_dir,'eses_subjects_202008.csv')

name_id_col = "BK_name"
group_id_col = "group"
msr_dict = {'cc':"clusterin coefficienct", 'pl':"path length",'ms':"mean strength",'mod':"modularity", 'le':"local efficiency"}
debug = ''

dt = datetime.today().strftime('%Y%m')

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
    parcel_dict = OrderedDict()
#    if network_name:
#        print(f'Selecting ROIs belonging to the {network_name} network.\n')
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

def create_conn_df(mdata, abs_thr=None, prop_thr=None, triu=False):
    """"Create the full, filterable connectivity matrix. Add subject id and group info"""
    subj_ix = mdata['Z'].shape[-1]
    x, subj_dict, group_dict = get_subj_df_data(subjects_file)
    if debug:
        print('Getting the whole connectivity matrix, so get_parcel_dict will return all ROIs intentionally.')
    parcel_dict = get_parcel_dict(mdata, network_name=None)
    rois = list(parcel_dict.keys())
    col_names = [name_id_col] + rois # Use the subj column to be able to search and filter by specific participants
    conn_df = pd.DataFrame(columns = col_names)
    abs_thr_dict = {}
    for s in range(0, subj_ix):
        tmp_df = pd.DataFrame(mdata['Z'][:,:mdata['Z'].shape[0],s], index = rois, columns = rois) # Grab the ROIs that are part of the atlas and not all the extra regressor correlations
        sign_mask = np.sign(tmp_df)
        tmp = tmp_df.abs().to_numpy(na_value=0)

        if abs_thr:
            try:
                tmp = bct.threshold_absolute(tmp, abs_thr, copy=False)
            except Exception as E:
                print(f'{E}') # Likely that the threshold input is not a float
            tmp[tmp == -0] = 0
            abs_thr_dict[subj_dict[s]] = np.count_nonzero(tmp)

        if prop_thr:
            #Second, proportionally threshold the matrix. May be used in combination with the absolutely thresholded values
            try:
                tmp = bct.threshold_proportional(tmp, prop_thr, copy=False)
            # There is a note in the BCT wiki about the behavior of this function. Specifically, being careful with matrices that are both signed and sparse. Thus, the input is the absolute value of the connectivities (for now).
            except Exception as E:
                print(f'Exception thrown from create_conn_df:\n{E}') # Likely that the threshold input is not a float

        if triu==True:
            tmp[np.triu_indices(tmp.shape[0], k=0)] = np.nan

        tmp_df = pd.DataFrame(data=tmp, index = rois, columns=rois)
        tmp_df = tmp_df*sign_mask
        tmp_df = tmp_df.replace([-0,np.nan],0)
        tmp_df[name_id_col] = subj_dict[s]
        conn_df = pd.concat([conn_df,tmp_df])

    conn_df[group_id_col] = conn_df[name_id_col].map(group_dict)
    if abs_thr_dict:
        conn_df['abs_thr_cxns'] = conn_df[name_id_col].map(abs_thr_dict)
    if debug:
        print(f'create_conn_df columns: {conn_df.columns}\nIndex: {conn_df.index}')
    return conn_df


def get_network_matrix(mdata, network_name=None, subj_list=None, abs_thr=None, prop_thr=None, triu=False):
    """Provides the overarching connectivity for all participants as a searchable dataframe. No group assignments or covariates are included by this method."""
    parcel_dict = get_parcel_dict(mdata, network_name)
    # Select the index for the third dimension fo the numpy array
    if subj_list:
        if not isinstance(subj_list,list):
            subj_list = [subj_list]

    # Apply filters to the connectivity dataframe
    conn_df = create_conn_df(mdata, abs_thr, prop_thr, triu)
    if subj_list:
        print(f'Gathering {subj_list}')
    if network_name:
        if abs_thr:
            cols = [col for col in conn_df.columns if col in ([name_id_col, group_id_col,'abs_thr_cxns'] + list(parcel_dict.keys()))]
        else:
            cols = [col for col in conn_df.columns if col in ([name_id_col, group_id_col] + list(parcel_dict.keys()))]
        conn_df = conn_df[cols][conn_df.index.isin(parcel_dict.keys())]

        nonzeros = int(((conn_df!=0).sum(1).sum())/(mdata["Z"].shape[-1]))
        print(f'{nonzeros} non-zero values exist in {network_name}')
    if subj_list:
        if isinstance(subj_list[0], int):
            conn_df = conn_df.iloc[subj_list,:] # Compatibility with numpy logic
        else:
            conn_df = conn_df.loc[conn_df[name_id_col].isin(subj_list),:]
        if debug:
            print(subj_list)
        #print(f'After filtering for {len(subj_list)} participants, the matrix is {conn_df.shape}\n')
    return conn_df


def get_cohort_network_matrices(mdata, network_name, group, mean=False, abs_thr=None, prop_thr=None, triu=False):
    ''' Get the matrices for a cohort of patients in a given network. '''
    if debug:
        print('get_cohort_network_matrices')
    parcel_dict = get_parcel_dict(mdata, network_name=None)
    rois = list(parcel_dict.keys())
    cohort_df = get_network_matrix(mdata,network_name, abs_thr = abs_thr, prop_thr=prop_thr)
    cohort_df = cohort_df.loc[cohort_df[group_id_col]==group,:]
    drop_cols = [col for col in cohort_df.columns if col not in rois]
    cohort_df.drop(columns = drop_cols, inplace = True)
    cols = cohort_df.columns
    print(f'After group filter, matrix size is {cohort_df.shape}')
    if mean is True:
        if debug:
            print(cohort_df.groupby(level=0).mean().reindex(cols))
        return cohort_df.groupby(level=0).mean().reindex(cols)
        #return np.nanmean(cohort_df.to_numpy(), axis=0) # This flattened the array and didn't seem to return the means by ROI
    else:
        return cohort_df


def check_subj_avlbl(subj):
    """"Figure out whether the given subject input is valid for the study."""
    x, subj_dict, y = get_subj_df_data()
    if subj not in subj_dict.values():
        return_status = 0
        print(f'Subject {subj} is not present in the sample.\nChoose from {subj_dict.values()} and re-run')
    else:
        return_status = 1
    return return_status

def plot_network_matrix(mdata, network_name, subj):
    if subj:
        status = check_subj_avlbl(subj)
    else:
        status = 1
    if status == 1:
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
    im = ax.matshow(mean_matrix.to_numpy())
    fig.colorbar(im)
    parcels = get_parcel_dict(mdata, network_name)
    plt.xticks(np.arange(len(parcels.keys())), list(parcels.keys()), rotation='vertical')
    plt.yticks(np.arange(len(parcels.keys())), list(parcels.keys()), rotation='horizontal')
    plt.show()

def plot_cohort_comparison(mdata, network_name, group_1, group_2, vmin=None, vmax=None):
    mean_matrix_1 = get_cohort_network_matrices(mdata, network_name, group_1, mean=True)  # need to collect all the matrices to add
    mean_matrix_1 = mean_matrix_1.to_numpy()
    mean_matrix_2 = get_cohort_network_matrices(mdata, network_name, group_2, mean=True)
    mean_matrix_2 = mean_matrix_2.to_numpy()
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

def get_cohort_graph_msr(mdata, network_list, prop_thr_list=None, msr_list=['cc'], update=False, positive_only=True):
    study_df_file = op.join(data_dir,dt+'_graph_msr.csv')
    if op.isfile(study_df_file) and op.getsize(study_df_file) > 0 and update==False:
        study_df = pd.read_csv(study_df_file)
        study_df = study_df.rename({'Unnamed: 0':'rois'}, axis=1)
        if not all(el in msr_list for el in study_df.columns):
            print(el in msr_list for el in study_df.columns)
            #update = True
    elif not op.isfile(study_df_file) or not op.getsize(study_df_file) > 0:
        print('File was empty. Re-creating.')
        update=True

    if update == True:
        study_df = pd.DataFrame(columns=['group','network','fc','prop_thr'] + msr_list)
        for prop_thr in prop_thr_list:
            for network in network_list:
                network_df = get_network_matrix(mdata, network_name=network, prop_thr=prop_thr)
                for subj in set(network_df[name_id_col]):
                    tmp = indvd_summary_and_graph_calcs(mdata, subj, network, msr_list=msr_list, positive_only=positive_only, prop_thr=prop_thr)
                    study_df = pd.concat([study_df,tmp])
        study_df = study_df.dropna(axis=1,how='all')
        study_df = study_df.drop_duplicates()
        study_df.sort_values([group_id_col, name_id_col], inplace=True)
        study_df.to_csv(study_df_file)
    return study_df

def indvd_summary_and_graph_calcs(mdata, subj, network, msr_list=['cc'], positive_only=True, prop_thr=None):
    """Individual graph measures are calculated and returned as a dataframe.
    This function is dependent on a population-wide DF being passed, rather than re-creating through get_network_matrix."""
    network_df = get_network_matrix(mdata, network_name=network, subj_list=[subj], prop_thr=prop_thr)
    parcel_dict = get_parcel_dict(mdata, network_name=network)
    rois = list(parcel_dict.keys())
    tmp = pd.DataFrame(index=rois)
    mat = network_df[rois].to_numpy(na_value=0) # Still has negatives
    np.fill_diagonal(mat,0) # BCT compatibility
    for msr in msr_list:
        if positive_only == True:
            mat = mat*(mat > 0)
            mat_sk = preprocessing.normalize(mat)
            mat_norm = bct.weight_conversion(mat, 'normalize')
        if positive_only == True and msr == 'cc':
            tmp[msr]  = bct.clustering_coef_wu(mat).tolist()
            tmp[msr+'_sk_norm'] = bct.clustering_coef_wu(mat_sk).tolist()
            tmp[msr+'_normed']  = bct.clustering_coef_wu(mat_norm).tolist()
        elif positive_only == True and msr == 'mod':
            tmp[msr] = bct.community_louvain(mat)[1].tolist() #get q values
            tmp[msr+'_sk_norm'] = bct.community_louvain(mat_sk)[1].tolist()
            tmp[msr+'_normed'] = bct.community_louvain(mat_norm)[1].tolist()
        elif msr== 'cc' :
            tmp[msr+'_pos'] = bct.clustering_coef_wu_sign(mat)[0].tolist()
            tmp[msr+'_neg'] = bct.clustering_coef_wu_sign(mat)[-1].tolist()
    tmp[name_id_col]  = network_df.loc[network_df[name_id_col]==subj, name_id_col]
    tmp[group_id_col] = network_df.loc[network_df[name_id_col]==subj, group_id_col]
    tmp['network'] = network
    tmp['prop_thr'] = prop_thr
    tmp['fc'] = network_df[rois].mean(axis=0)
    tmp['fc_sk_norm'] = np.mean(mat_sk,axis=0).tolist()
    tmp['fc_bctpy_normed'] = np.mean(mat_norm, axis=0).tolist()

    return tmp


def assess_summaries_and_graph_calcs(mdata,subj_list=None, network_list=None, msr_list=['cc'], prop_thr_list=None, positive_only=True):
    study_df = get_cohort_graph_msr(mdata, network_list=network_list, prop_thr_list=prop_thr_list, msr_list=msr_list, update=False, positive_only=True)
    msr_dict = {msr:['mean','std','max','min'] for msr in study_df.columns if msr not in ['group', name_id_col, group_id_col, 'network', 'rois','prop_thr']}
    #print(list(msr_dict.items()))
    # Aggregrate the subject-level
    plot_df = study_df.groupby([name_id_col,'network']).agg(msr_dict)
    print(plot_df)

    for network in network_list:
        for msr in sorted(msr_dict.keys()):
            fig, ax = plt.subplots(figsize=(20,6))
            if "prop_thr" in study_df.columns:
                study_df['prop_thr']=study_df['prop_thr'].astype('category')
                sns.stripplot(x=name_id_col, y=msr, hue="prop_thr", dodge=True, data=study_df.loc[study_df['network']==network,:], jitter=True)
            else:
                sns.stripplot(x=name_id_col, y=msr, data=study_df.loc[study_df['network']==network,:], jitter=True)
            plt.title(network)
            plt.xticks(rotation=80)
            plt.show()
    return study_df, plot_df

def calculate_AUC(mdata, bootstrap=5000, subj_list=None, network_list=None, msr_list=['cc'], prop_thr_list=[0,.5,1], positive_only=True, update=False):
    study_df = get_cohort_graph_msr(mdata, network_list=network_list, prop_thr_list=prop_thr_list, msr_list=msr_list, update=update, positive_only=True)
    msr_dict = {msr:['mean','std','max','min'] for msr in study_df.columns if (msr not in ['group', name_id_col, group_id_col, 'network', 'rois','prop_thr']) and any(m in msr for m in msr_list)}
    agg_df = study_df.groupby([name_id_col,'network']).agg(msr_dict)

    for network in set(study_df['network']):
        for msr in msr_dict.keys():
            print(msr.upper())
            prmt_rslts = pd.DataFrame() # for permutation results
            tmp = study_df[study_df['network'] == network]
            tmp.drop_duplicates(inplace=True)
            groups = list(set(tmp[group_id_col]))
            auc_dict = {}
            for g,group in enumerate(groups):
                group_mean = []
                for subj in set(tmp.loc[tmp[group_id_col]==group,name_id_col]):
                    group_mean.append(auc_helper(tmp.loc[(tmp[group_id_col]==group) & (tmp[name_id_col]==subj),:],'prop_thr',msr))
                auc_dict[g] = np.mean(np.array(group_mean))
                print(f'Avg AUC for {group}={np.mean(np.array(group_mean))}')
            auc_diff = auc_dict[0]-auc_dict[1]
            size_grp2 = len(set(tmp.loc[tmp[group_id_col]==groups[1],name_id_col]))

            print(f'Group 1 has {len(set(tmp.loc[tmp[group_id_col]==groups[0],name_id_col]))} people.')
            print(f'Group 2 has {size_grp2} people.')

            for c in tqdm(range(1,bootstrap), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
                permute_tmp_grp2 = random.sample(set(tmp[name_id_col]), size_grp2)
                permute_tmp_grp1 = [subj for subj in set(tmp[name_id_col]) if subj not in permute_tmp_grp2]
                if c % 500 == 0:
                    print(permute_tmp_grp2)
                group_mean = []
                for g, group in enumerate([permute_tmp_grp1, permute_tmp_grp2]) :
                    for subj in group:
                        try:
                            group_mean.append(auc_helper(tmp.loc[tmp[name_id_col]==subj,:],'prop_thr',msr))
                        except Exception as e:
                            print(f'{subj} failed')
                            print(e)
                    prmt_rslts.loc[c,'test'] = c
                    prmt_rslts.loc[c,'auc_grp'+str(g+1)] = np.mean(np.array(group_mean))
                    #print(f'Group {g} mean = {np.mean(np.array(group_mean))}')

            prmt_rslts['auc_diff'] = prmt_rslts['auc_grp1'] - prmt_rslts['auc_grp2']
            prmt_rslts = prmt_rslts.sort_values('auc_diff').reset_index()
            print(f"The experimental AUC difference, {auc_diff.round(3)}, is {prmt_rslts.loc[prmt_rslts['auc_diff'] >= auc_diff].index[0]/bootstrap} through the boostrapped results.")
#            print(prmt_rslts.index(prmt_rslts['test']=='experimental').tolist())

            fig,ax = plt.subplots()
            sns.histplot(prmt_rslts, x='auc_diff', kde=True)
            plt.axvline(auc_diff, color='r',linewidth=5)
            plt.title(f'{network}:{msr}')
            plt.show()

def auc_helper(individ_df,x_axis, y_axis):
    individ_df = individ_df.groupby(x_axis).agg({y_axis:'mean'}).reset_index()
    x = individ_df.index.values
    y = individ_df[y_axis]
    return metrics.auc(x,y)

def plot_range_of_thresholds(mdata, network_list, prop_thr_list=None, msr_list=["cc"]):
    """Test and plot a range of thresholds to see how thresholds may affect hypothesis testing between two groups."""
    if not isinstance(msr_list,list):
        msr_list = [msr_list]
    study_df = get_cohort_graph_msr(mdata, network_list, prop_thr_list=prop_thr_list, msr_list=msr_list)
    study_df = study_df.dropna(axis=1,how='all')
    #msrs_to_agg = {col:['mean','std','count'] for col in study_df.columns if msr for msr_list in col }
    #print(study_df.groupby(['network', 'group', 'prop_thr']).agg(msrs_to_agg))
    msr_list = [col for col in study_df.columns if any(msr in col for msr in msr_list) ]
    result_dict = scores_by_network(mdata, network_list, msr_list=msr_list, study_df=study_df, prop_thr_list = prop_thr_list)

    for n in set(study_df['network']):
        for msr in msr_list:
            f,ax = plt.subplots()
            sns.lineplot(x='prop_thr',y=msr,hue='group', palette='tab10',data=study_df.loc[study_df['network']==n,:], ci=95)
            #sns.lineplot(x='prop_thr',y='cc_negConn',palette='winter',hue='group',data=study_df.loc[study_df['network']==n,:], ci='sd')
            plt.title(n.title())
            plt.xlabel('Proportional thresholding value')
            plt.ylabel(f'Mean {msr}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            print(f'ks_stat: {round(result_dict[msr][n].statistic,4)}')
            print(f'pvalue: {result_dict[msr][n].pvalue}')

def plot_range_of_thresholds_individs(mdata, network_list, prop_thr_list=None, msr_list=['cc']):
    study_df = get_cohort_graph_msr(mdata, network_list, prop_thr_list=prop_thr_list)
    msrs_to_agg = {col:['mean','std','count'] for col in study_df.columns if msr for msr_list in col }

    print(study_df.groupby(['network', 'group', 'prop_thr']).agg(msrs_to_agg))
    for n in set(study_df['network']):
        for msr in msrs_to_agg.keys():
            f,ax = plt.subplots()
            for subj in set(study_df[name_id_col]):
                if set(study_df.loc[(study_df['network']==n) & (study_df[name_id_col]==subj),group_id_col]) == 'hc':
                    sns.lineplot(x='prop_thr',y=msr,color='red',data=study_df.loc[(study_df['network']==n) & (study_df[name_id_col]==subj),:])
                else:
                    sns.lineplot(x='prop_thr',y=msr,color='blue',data=study_df.loc[(study_df['network']==n) & (study_df[name_id_col]==subj),:])
            plt.title(n.title())
            plt.xlabel('Proportional thresholding value')
            plt.ylabel(f'Mean {msr_dict[msr]}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

def scores_by_network(mdata, network_list, msr_list, study_df=None, prop_thr_list=None, positive_only=False):
    if study_df.empty:
        study_df = get_cohort_graph_msr(mdata, network_list, prop_thr_list=prop_thr_list, msr_list = msr_list, positive_only=positive_only)
    groups = list(set(study_df[group_id_col]))
    result_dict = defaultdict(dict)
    for network in network_list:
        if network not in study_df['network'].unique():
            print(f'Need to rebuild the df as {network} does not exist in the the current one.')
            study_df = get_cohort_graph_msr(mdata, network_list, prop_thr_list=prop_thr_list, msr_list = msr_list, update = True, positive_only=positive_only)
        for msr in msr_list:
            result_dict[msr][network] = scipy.stats.ks_2samp(study_df.loc[((study_df[group_id_col] == groups[0]) & (study_df['network'] == network)), msr], study_df.loc[((study_df[group_id_col] == groups[1]) & (study_df['network'] == network)), msr])
    return result_dict


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
