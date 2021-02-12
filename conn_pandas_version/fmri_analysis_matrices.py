"""
Functions to explore connectivity matrices generated through CONN.

Date: August 2020
@author: Josh Bear, MD
v0.2 (BMS) Adapted fmri_analysis_functions for pandas
"""
import os
from glob import glob
import numpy as np
import pandas as pd
import json
from collections import OrderedDict, defaultdict
from datetime import datetime
import multiprocessing
import itertools

import fmri_analysis_utilities as utils
import fmri_analysis_load_funcs as faload

config = faload.load_config()

class analysis():
    def __init__(self,network_name, mdata=None, prop_thr=0, subject_list=None, triu=True,  wb_norm=False, abs_val=False):
        if mdata:
            self.mdata = mdata
        else:
            self.mdata = config.mdata
        self.network = network_name
        self.prop_thr = prop_thr
        self.subject_list = subject_list
        self.triu = triu
        self.wb_norm = wb_norm
        parcel_dict = faload.get_parcel_dict(self.mdata, network_name=self.network)
        self.rois = list(parcel_dict.keys())
        self.abs_val = abs_val

    def create_master_conn_df(self):
        """"Create the full, filterable connectivity matrix with subject id and group info."""
        subj_ix = self.mdata['Z'].shape[-1]
        x, subj_dict, group_dict = faload.get_subj_df_data(config.nonimaging_subjectlevel_data)
        all_rois = faload.get_parcel_dict(self.mdata, network_name=None) # For the master df, get ALL the rois, not just the network-specific ones. Filter later.
        col_names = [config.name_id_col] + list(all_rois.keys()) # Use the subj column to be able to search and filter by specific participants
        master_conn_df = pd.DataFrame(columns = col_names)
        data_dfs = []
        for s in range(0, subj_ix):
            tmp_df = pd.DataFrame(self.mdata['Z'][:,:self.mdata['Z'].shape[0],s], index = all_rois, columns = all_rois)
            tmp_df[config.name_id_col] = subj_dict[s]
            data_dfs.append(tmp_df)
        master_conn_df = pd.concat(data_dfs)
        self.master_conn_df = master_conn_df.reset_index()

        if self.wb_norm:
            self.master_conn_df = utils._normalize(self.master_conn_df)       
        setattr(self,'curr_df', master_conn_df.reset_index())

    def produce_matrix(self):
        if not hasattr(self,'curr_df'):
            print('Creating the master df')
            self.create_master_conn_df()
        if self.network:
            if not self.rois == set(self.curr_df.index):
                self.curr_df = utils.filter_conn_df_network(self.curr_df,self.rois)
        if self.subject_list:
            if not self.subject_list == set(self.curr_df[config.name_id_col]):
                self.curr_df = utils.filter_conn_df_subjects(self.curr_df,self.subject_list)
        if self.prop_thr:
            if self.abs_val:
                self.curr_df = utils._abs_val_thr(self.curr_df, self.prop_thr)
            else:
                self.curr_df = utils._threshold(self.curr_df, self.prop_thr)
        if self.triu:
            self.curr_df = utils._triu(self.curr_df)

        return self.curr_df

    def update_thr(self, prop_thr=None):
        setattr(self,'prop_thr',prop_thr)
        return self.produce_matrix()

    def update_network(self,network=None):
        rois = faload.get_parcel_dict(self.mdata, network_name=network)
        setattr(self,'network',network)
        setattr(self,'rois',rois)
        return self.produce_matrix()

# TODO make a separate module to calculate graph metrics. The output may be appended to the master df.
