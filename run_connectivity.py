# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:13:10 2020

@author: Brianne
"""


from scipy.io import loadmat
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from importlib import reload
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import sys


import fmri_analysis_functions_pd as faf

class network_analysis():
    def __init__(self, main_dir, conn_dir,subjects_file, data_dir, conn_file):
        self.main_dir = main_dir
        self.conn_dir = conn_dir
        self.subjects_file = subjects_file
        self.data_dir = data_dir
        self.conn_file = conn_file
        print('Importing the mdata matrix will take a while.')
        self.mdata, rois = faf.load_mat(data_dir, conn_file)
        self.avlbl_network_list = set([roi.split('_')[0] for roi in rois])


    def gather_input_options(self):
        """Ask the user if there are specific cutoffs and other options for this specific analysis."""
        self.vmax = input('For graphing comparisons, what would you like vmax to be?')
        self.vmin = input('For graphing comparisons, what would you like vmin to be?')

    def gather_data_and_dicts(self, group1, group2, network_name=None):
        self.network_name = network_name
        self.cv_df, self.subjs_dict, self.groups_dict = faf.get_subj_df_data(self.subjects_file) #cv for covariate
        self.group1_ix = faf.get_group_membership(self.cv_df, group1)
        self.group2_ix = faf.get_group_membership(self.cv_df, group2)


    def characterize_networks(self, subj=None, group1=None, group2=None, network_name=None):
        if not hasattr(self,'cv_df'):
            self.gather_data_and_dicts(group1, group2, network_name)
        if subj:
            faf.plot_network_matrix(self.mdata, network_name, subj)
        elif group1 and not group2:
            faf.get_cohort_network_matrices(self.mdata, network_name, group1)
        elif group2:
            if not self.vmin:
                self.gather_input_options()
            faf.plot_cohort_comparison(self.mdata, network_name, group1, group2, vmax=self.vmax, vmin=self.vmin)
            faf.describe_cohort_networks(self.mdata, network_name)


    def get_network_df_for_bct(self, network_name, group=None, subj=None):
        if subj:
            df = faf.get_network_matrix(self.mdata, network_name = network_name, subj_list=subj)
            df.drop(columns=['BK_name','group'],inplace=True)
        elif group:
            df = faf.get_cohort_network_matrices(self.mdata, network_name, group, mean=True)
        else:
            print('Either group or subject is required to be able to grab the correct data.')
        rois = df.columns
        df = df.to_numpy()
        np.fill_diagonal(df, 0)
        return df, rois
    
    def compare_groups(self, group_list, network_list=None, abs_thr_list=None, prop_thr_list=None):
        x, nodes = self.get_network_df_for_bct(network_list[0],group_list[0])
        df = pd.DataFrame(index=list(itertools.chain.from_iterable(itertools.repeat(x,len(group_list)*len(network_list)) for x in nodes)),columns = ['group','network','abs_thr','prop_thr','cc'])
        
        for group in group_list:
            for network in network_list:
                for abs_thr in abs_thr_list:
                    for prop_thr in prop_thr_list:
                        tmp, x = self.get_network_df_for_bct(network,group, abs_thr=abs_thr, prop_thr=prop_thr)
                        np.fill_diagonal(tmp,0)
                        tmp1=pd.DataFrame()
                        tmp1['cc'] = bct.clustering_coef_wu(tmp).tolist()
                        tmp1['group'] = group
                        tmp1['network'] = network
                        tmp1['abs_thr'] = abs_thr
                        tmp1['prop_thr'] = prop_thr

                        try:
                            df = pd.concat([df,tmp1])
                        except:
                            pass
        print(df.groupby(['group','network','abs_thr','prop_thr']).agg({'cc':['mean','std','count']}))
        return df

            
            

if __name__ == "__main__":
    na1 = network_analysis(main_dir, conn_dir,subjects_file, data_dir, conn_file)
    na1.characterize_networks(network_name='att', group1='hc', group2='eses')
