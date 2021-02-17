import pandas as pd
import numpy as np
import fmri_analysis_get_data as get
from importlib import reload
import os, glob
from datetime import datetime


def check_data_loaded():
    import shared
    if not hasattr(shared, 'group1_indices'):
        get.get_subj_df_data()
        reload(shared)


def drop_negatives(matrix):
    matrix[matrix < 0] = np.nan
    return matrix


def filter_df(df, criteria={}):
    df.dropna(how='all', inplace=True)
    # Other criteria as we build out the pipelines
    return df


def match_subj_group(subj_ix):
    """ Match subject index and group. May be obsolete in favor of
        subject_converter. """
    import shared
    if subj_ix in shared.group1_indices:
        return shared.group1
    elif subj_ix in shared.group2_indices:
        return shared.group2
    else:
        return np.nan


def parallel_setup():
    """Calculate the number of threads that should be included in the pool."""
    import multiprocessing as mp
    import math
    cores = mp.cpu_count()
    if cores <= 8:
        job_limit = math.ceil(.25*cores)
    else:
        job_limit = math.ceil(.3*cores)  # Play nice with other super users.
    return mp.Pool(job_limit)


def roiIx_to_name_translator():
    return faload.load_network_parcels()


def save_df(df, filename):
    import shared
    now = datetime.now().strftime('%y%m%d%H%M')
    out_filename = os.path.join(shared.main_dir, str(now)+filename)
    df.to_csv(out_filename, index=False)


def subject_converter(df, orig_subj_col='subj',
                      add_characteristics=['subject', 'group']):
    import shared
    demos = pd.DataFrame(pd.read_csv(shared.nonimaging_subjectlevel_data))
    demos.reset_index(inplace=True)
    df = df.merge(demos[add_characteristics+['index']],
                  left_on=orig_subj_col, right_on='index')
    return df


def nan_bouncer(x, axis=0):
    # https://stackoverflow.com/questions/48101388/remove-nans-in-multidimensional-array
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
    mask = np.isnan(x)
    cut = np.min(np.count_nonzero(mask, axis=0))
    idx = tuple(np.ogrid[tuple(map(slice, x.shape[1:]))])
    res = x[(np.argsort(~mask, axis=0, kind='mergesort')[cut:],) + idx]
    return res if axis == 0 else np.moveaxis(res, 0, axis)

def check_msrs_calcd(msrs_needed, prop_thr,csv_file_pattern='*csv'):
    """
    Locates previously calculated and saved results to avoid long
    re-processing times and compute.

    Parameters
    ----------
    msrs_needed : list
        Each measure in this list will be searched for in the column
        names to return matches.

    prop_thr : float
        Methods calling this one should have a specific threshold that
        is associated with the calculations. Use that same threshold
        to cross-check previous analyses.
    
    csv_file_pattern (optional) : filename
        Helpful to zero in on previously saved results, particularly if
        a certain result is desired to check against.

    Returns
    -------
    calcd_msrs : dict
    """

    import shared
    import collections
    # Create expandable, nested dictionaries
    avlbl_dict = collections.defaultdict(dict)
    csv_locs = glob.glob(os.path.join(shared.proj_dir, csv_file_pattern))
    # Get the measures that are available and at a given threshold

    # Compare them against the requested measures

    # Output a dictionary of measures, thresholds, and values.