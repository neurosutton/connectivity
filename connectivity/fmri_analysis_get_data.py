"""
Functions to explore connectivity matrices generated through CONN.

Date: August 2020
@author: Josh Bear, MD with organization by Brianne Sutton, PhD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""
import os
import numpy as np
import fnmatch
import pandas as pd
from scipy.io import loadmat
import fmri_analysis_utilities as utils
import fmri_analysis_load_funcs as faload
import fmri_analysis_plotting as faplot
import fmri_analysis_manipulations as fam
import shared


def get_mdata(conn_dir=None, conn_file=None):
    """ Loading and reloading the module is much quicker with loading
        the matrix as its own method. Call first, so that there is data,
        though. """
    if not conn_dir:
        conn_dir = shared.conn_dir  # get_conn_dir
        conn_file = shared.conn_file  # get_conn_file
    return loadmat(os.path.join(conn_dir, conn_file))


def get_conn_data(mdata=None, roi_count=None, clear_triu=True, subset=[]):
    """Foundational method to transform the MATLAB matrices to numpy matrices.
    Output:
    mdata     : loaded in the shared object; dictionary of arrays in line with
                MATLAB data structures.
    conn_data : Just the connectivity matrices extracted from mdata;
                square matrix excludes the regressors and atlas-based values
                that CONN adds to the right side of the matrix
    """
    mdata = get_mdata() if mdata is None else mdata
    roi_count = mdata['Z'].shape[0] if roi_count is None else roi_count
    conn_data = mdata['Z'][:roi_count, :roi_count, :]
    if subset:
        if not isinstance(subset, list):
            subset = [subset]
        if set(subset).issubset([shared.group1, shared.group2]):
            if not hasattr(shared, 'group1_indices'):
                get_subj_df_data()
            key = [
                k + "_indices" for k,
                v in shared.__dict__.items() if v == subset[0]][0]
            subset = getattr(shared, key)
            print(f'Subset has {len(subset)} participants.')
        conn_data = conn_data[:, :, subset]
        print(f'Connectivity matrix = {conn_data.shape}')
    if clear_triu:
        for subject in range(conn_data.shape[2]):
            conn_data[:, :, subject][np.triu_indices(
                conn_data.shape[0], 0)] = np.nan
    return conn_data


def get_network_parcels(network, mdata=None):
    """Returns parcel names and indices with HCP remaining in the
    name and indexed to work with numpy-based functions.
    Output: {atlas_name.roi (str): numpy index of ROI (int)}
    """
    mdata = get_mdata() if mdata is None else mdata

    parcel_names = [str[0].lower() for str in mdata['names'][0]]
    parcels = {k.split('.')[-1]: v for v, k in enumerate(parcel_names)}
    if network and network not in ["wb", "whole_brain", "whole brain"]:
        pattern = network.lower() + '*'
        matching = fnmatch.filter(parcels.keys(), pattern)
        network_parcels = {k: v for k, v in parcels.items() if k in matching}
    else:
        network_parcels = parcels

    return network_parcels


def get_subj_df_data(nonimaging_subjectlevel_data=None):
    """Primarily for reading in demographic and neuropsychological data."""
    # TODO make more robust to reorganization from editor. Parse "conn_name"
    # Subject001 and match MATLAB/python indexing to make sure the correct
    # participants are identified.
    nonimaging_subjectlevel_data = shared.nonimaging_subjectlevel_data if nonimaging_subjectlevel_data is None else nonimaging_subjectlevel_data
    subj_df = pd.DataFrame(pd.read_csv(nonimaging_subjectlevel_data))
    subj_df = utils.filter_df(subj_df)
    subj_dict = {k: v for k, v in enumerate(subj_df[shared.name_id_col])}
    group_dict = dict(
        zip(subj_df[shared.name_id_col], subj_df[shared.group_id_col]))
    grp1_indices = [i for i, x in enumerate(
        list(subj_df[shared.group_id_col])) if x == shared.group1]
    grp2_indices = [i for i, x in enumerate(
        list(subj_df[shared.group_id_col])) if x == shared.group2]
    shared.group1_indices = grp1_indices
    shared.group2_indices = grp2_indices
    faload.update_shared(shared)
    return subj_df, subj_dict, group_dict


def subj_data():
    # Alias to eliminate the "extra" variables.
    subj_df, x, x = get_subj_df_data()
    return subj_df


def get_subject_scores(measure):
    """Gather cognitive or medical scores."""
    scores = {}
    scores_df = pd.DataFrame(columns=['index', 'subject', measure])
    subj_data, x, x = get_subjget_subj_df_data()

    for row in subj_data.index:
        if not np.isnan(float(subj_data[subj_data.index == row][measure])):
            scores_df.loc[len(scores_df)] = [row,
                                             str(subj_data[subj_data.index == row]['subject'].values[0]),
                                             float(subj_data[subj_data.index == row][measure])]
            # scores[row] = float(subj_data[subj_data.index == row][measure])
    return scores_df


def get_network_matrix(
        network_name,
        subj_idx,
        conn_data=None,
        prop_thr=None,
        network_mask=None,
        exclude_negatives=False,
        normalize=False):
    '''
    Adding a normalize, which can call different types.
        - 'self' will divide by own whole brain mean connectivity
    '''
    utils.check_data_loaded()
    prop_thr is None if prop_thr == 0 else prop_thr
    conn_data = get_conn_data() if conn_data is None else conn_data
    parcels = get_network_parcels(network_name)
    indices = list(parcels.values())
    matrix = conn_data[:, :, subj_idx][np.ix_(indices, indices)]
    if prop_thr or network_mask is not None:
        if prop_thr:
            network_mask = fam.make_proportional_threshold_mask(
                network_name=network_name,
                prop_thr=prop_thr,
                exclude_negatives=exclude_negatives)
        matrix = network_mask * matrix
        matrix[matrix == 0] = np.nan
    if normalize is not False:
        # for start, will just assume it's 'self'
        # Does this matrix need to be sorted per "indices"
        self_norm_value = np.nanmean(
            utils.drop_negatives(conn_data[:, :, subj_idx]))
        matrix = matrix / np.absolute(self_norm_value)
    return matrix


def get_cohort_network_matrices(
        network_name,
        subj_idx,
        mean=False,
        conn_data=None,
        prop_thr=None,
        subject_level=False,
        network_mask=None,
        exclude_negatives=False):
    """Output:
       When no mean or subject_level args are selected, array of matrices with the thresholded network connectivity values. Dimensions are (num_of_subjs,ROIs, ROIs)
       When mean is True, the array of matrices becomes 2D mean of ROI connectivity.
       When subject_level is True (subservient to the mean arg), matrix is subject as row and grand mean of connectivities as the single column.
       When subject_level AND mean are selected, the final matrix is subject by mean ROI.
       """
    conn_data = get_conn_data() if conn_data is None else conn_data

    cohort_matrices = []  # need to collect all the matrices to add
    for subj in subj_idx:
        matrix = get_network_matrix(
            network_name,
            subj,
            conn_data=conn_data,
            prop_thr=prop_thr,
            network_mask=network_mask,
            exclude_negatives=exclude_negatives)
        cohort_matrices.append(matrix)
    cohort_matrices = np.asarray(cohort_matrices)
    if mean and subject_level:
        return np.nanmean(cohort_matrices, axis=2)
    if mean is True:
        return np.nanmean(cohort_matrices, axis=0)
    elif subject_level is True:
        return np.nanmean(cohort_matrices, axis=(1, 2))
    else:
        return cohort_matrices


def get_cohort_comparison_over_thresholds(
        network_name,
        thr_range=None,
        thr_increment=None,
        conn_data=None,
        subject_level=False,
        plot=False,
        exclude_negatives=False):
    conn_data = get_conn_data() if conn_data is None else conn_data
    thr_increment = 0.1 if thr_increment is None else thr_increment
    thr_range = np.arange(
        0., 1, thr_increment) if thr_range is None else thr_range
    group_names = [shared.group1, shared.group2]
    comp_df = pd.DataFrame(columns=['threshold', 'group', 'connectivity'])
    df_idx = 0
    for value in thr_range:
        print(f'Working on {value}')
        network_mask = fam.make_proportional_threshold_mask(
            network_name, value, exclude_negatives=exclude_negatives)
        matrix_1 = get_cohort_network_matrices(
            network_name,
            shared.group1_indices,
            subject_level=subject_level,
            conn_data=conn_data,
            network_mask=network_mask)
        matrix_2 = get_cohort_network_matrices(
            network_name,
            shared.group2_indices,
            subject_level=subject_level,
            conn_data=conn_data,
            network_mask=network_mask)
        for conn in matrix_1.flatten():
            if not np.isnan(conn):
                comp_df.loc[df_idx] = [value, group_names[0], conn]
                df_idx = df_idx + 1
        for conn in matrix_2.flatten():
            if not np.isnan(conn):
                comp_df.loc[df_idx] = [value, group_names[1], conn]
                df_idx = df_idx + 1
    # fixes a potential rounding error in np.arange
    comp_df = comp_df.round(decimals={'threshold': 2})
    if plot:
        faplot.plot_cohort_comparison_over_thresholds(
            network_name, comp_df, group_names)
    return comp_df
