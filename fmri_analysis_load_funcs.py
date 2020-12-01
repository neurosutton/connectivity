"""
Functions to explore connectivity matrices generated through CONN.

Date: August 2020
@author: Josh Bear, MD
v0.2 (BMS) Adapted fmri_analysis_functions for pandas
"""
from scipy.io import loadmat
import scipy.stats
import matplotlib.pyplot as plt
import os
from glob import glob
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
from tqdm import tqdm
import multiprocessing
import itertools

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'directory_defs.json')) as f:
    defs = json.load(f)
    conn_dir = defs['conn_dir']
    main_dir = defs['main_dir']
data_dir = os.path.join(conn_dir)
conn_file = 'resultsROI_Condition001.mat'
nonimaging_subjectlevel_data =  os.path.join(main_dir,'eses_subjects_202008.csv')

name_id_col = "BK_name"
group_id_col = "group"
msr_dict = {'cc':"clustering coefficienct", 'pl':"path length",'ms':"mean strength", 'mod':"modularity", 'le':"local efficiency"}
debug = ''

dt = datetime.today().strftime('%Y%m')

class config(name_id_col, group_id_col, data_dir, conn_dir, dt):
    self.name_id_col = name_id_col
    self.group_id_col = group_id_col
    self.data_dir = data_dir
    self.conn_dir = self.conn_dir
    self.date = dt
    self.mdata,x = load_mat(self.data_dir, os.path.join(self.conn_dir,conn_file))

def load_mat(data_dir, conn_file):
    """Loading and reloading the module is much quicker with loading the matrix as its own method. Call first, so that there is data, though."""
    mdata = loadmat(os.path.join(data_dir, conn_file))
    rois = mdata['names']
    rois = [roi[0].replace('hcp_atlas.','') for r_array in rois for roi in r_array]
    return mdata, rois


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
    try:
        parcel_names = [str[0] for str in mdata['names'][0]]
    except:
        print(f'Issue in get_parcel_dict(mdata,network_name={network_name}, inverse={inverse})')
        print(mdata['names'])

    parcel_dict = OrderedDict()
#    if network_name:
#        print(f'Selecting ROIs belonging to the {network_name} network.\n')
    for p,parcel in enumerate(parcel_names):
        parcel = parcel.replace('hcp_atlas.','') # Clean the names
        if network_name and ('whole' not in network_name):
            if network_name.lower() in parcel.lower():
                parcel_dict[parcel] = p
            elif debug:
                print(f'Did not find {network_name} in {parcel}')
            else:
                pass
        else:
            parcel_dict[parcel] = p
    if inverse:
        parcel_dict = {v:k for k,v in parcel_dict.items()}
    if debug:
        print(f'Search "{network_name}" returned these ROIs:\n{parcel_dict}')
    return parcel_dict