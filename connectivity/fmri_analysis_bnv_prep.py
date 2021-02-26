"""
Create files for BrainNetViewer (BNV) visualization of network connectivity.

Date: December 2020
@author: Brianne Sutton, PhD
v0.1
"""
import pandas as pd
import numpy as np
import os
from importlib import reload

import fmri_analysis_utilities as utils
import fmri_analysis_manipulations as fam
import fmri_analysis_load_funcs as faload
import fmri_analysis_get_data as get
import shared

utils.check_data_loaded()

class bnv_analysis():
    def __init__(self, network=None, label_file=os.path.join(shared.atlas_dir,'hcpmmp1_expanded_labels.csv'), group=None, atlas_label="SuttonLabel", subject_list=None, prop_thr=0.7, prop_thr_pop='sample',exclude_negatives=shared.excl_negatives):
        self.network = "wb" if network is None else network
        self.group = group #Filtered group for the node values
        self.atlas_label = atlas_label
        self.label_df = pd.DataFrame(pd.read_csv(label_file))
        self.clean_labels()
        self.subject_list = subject_list
        self.prop_thr = prop_thr
        self.exclude_negatives = exclude_negatives
        if prop_thr_pop == 'sample':
           self.prop_thr_pop = []  # Can be 'sample' for the whole popultion or anything else to trigger self.group as the input
        elif self.subject_list:
           self.prop_thr_pop = self.subject_list
        else:
           self.prop_thr_pop = self.group

    def clean_labels(self):
        """Reduce mismatches and extraneous information from label file, so that the bare minimum needed for BNV is merged."""
        self.label_df = self.label_df[[
            'x', 'y', 'z', 'NETWORK', self.atlas_label]]
        self.label_df[self.atlas_label] = self.label_df[self.atlas_label].str.lower()

    def limit_labels(self, network=None):
        """Callable to reduce the labels and x,y,z coordinates to a network of interest."""
        if self.label_df.shape[1] > 4:
            self.clean_labels()
        parcels = get.get_network_parcels(self.network)
        indices = list(parcels.values())

        return self.label_df.loc[self.label_df[self.atlas_label].str.lower().isin(parcels)]
   
    def load_summary_data(self,analyze_group=None, bnv_node=True):
        """Allows string input for group comparison"""
        compare='no' # Tag for whether to subtract mean df's
        analyze_group = self.group if analyze_group is None else analyze_group

        if not analyze_group:
            grp_dict = {'shared.group1':shared.group1,'shared.group2':shared.group2}
        else:
            print('Looking for matching groups')
            if not isinstance(analyze_group,list):
                analyze_group = [analyze_group]
            if len(analyze_group) > 1:
                compare='yes' # more than one group
            grp_dict = {k:v for k,v in shared.__dict__.items() if str(v) in analyze_group}

        dfs = []
        for k in grp_dict.keys():
            indices = shared.__dict__[k.split('.')[-1]+'_indices'] # Flexible solve for group 1 or 2, depending on the group id from __init__
            print(indices)
            df = self.get_cohort_bnv_data(indices = indices, mean=bnv_node)
            df['group'] = grp_dict[k]
            dfs.append(df)
        df = pd.concat(dfs)
        if bnv_node:
            df = df.reset_index().rename(columns={'index':'rois'})
        if compare == 'yes':
            if 'subj_num' in df.columns:
                df = df.drop(columns=['subj_num'])
            df = self.calc_diff_df(df)
        else:
            df = df.groupby('rois').mean().reset_index().rename(columns={'index':'rois'})
        return df


    def get_cohort_bnv_data(self, indices=[1], mean=False):
        network_mask = fam.make_proportional_threshold_mask(self.network, self.prop_thr, exclude_negatives=self.exclude_negatives, subset = self.prop_thr_pop)
        parcels = get.get_network_parcels(self.network)
        subj_dfs =[]
        data = get.get_cohort_network_matrices(self.network, indices, mean=False, conn_data=None, prop_thr=self.prop_thr, subject_level=False, network_mask=network_mask, exclude_negatives=self.exclude_negatives)
        for subj in range(0,data.shape[0]):
            s = pd.DataFrame(data[subj], columns=list(parcels.keys()))
            s['subj_num'] = indices[subj]
            s['rois'] = list(parcels.keys())
            if mean:
                s = pd.DataFrame(s.groupby('rois').mean().mean(),columns=['fc'])
                s['subj_num'] = indices[subj]
        subj_dfs.append(s)
        return pd.concat(subj_dfs)


    def calc_diff_df(self, orig_df):
        """Create a difference dataframe between two groups."""
        groups = list(set(orig_df['group']))
        df1 = orig_df.loc[orig_df['group']==groups[0],:].drop(columns=['group']).groupby('rois').mean()
        df2=orig_df.loc[orig_df['group']==groups[1],:].drop(columns=['group']).groupby('rois').mean()
        df = df1-df2
        df['group'] = groups[0]+'-'+groups[1]
        df = df.reset_index()
        return df

    def make_node_file(self, msr_of_int='fc', analyze_group=None):
        analyze_group = self.group if analyze_group is None else analyze_group
        fc_df = self.load_summary_data(analyze_group=analyze_group, bnv_node=True)
        node_df = fc_df[[msr_of_int] + ['rois']]
        out_df = self.label_df.merge(node_df, left_on = self.atlas_label, right_on = 'rois').drop(columns=([self.atlas_label]))
        out_df['size'] = out_df[msr_of_int]
        out_df = out_df.apply(lambda x: x.str.strip() if x.dtype=="object" else x)
        out_df = out_df[['x','y','z',msr_of_int,'size','rois']]
        out_df.drop_duplicates(inplace=True)
        out_df = out_df.replace({np.nan:0})
        out_df.to_csv(os.path.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + ''.join(self.group) + '_' + str(self.prop_thr).split('.')[-1] + '_bnv.node'), header=False, index=False,sep='\t')


    def make_edge_file(self, analyze_group=None, mdata=None): 
        analyze_group = self.group if analyze_group is None else analyze_group
        mdata = get.get_mdata() if mdata is None else mdata
        parcels = get.get_network_parcels(self.network, mdata=mdata)
        indices = list(parcels.values())
        edges = self.load_summary_data(analyze_group=analyze_group, bnv_node=False)
        drop_cols = edges.columns.intersection([shared.name_id_col, shared.group_id_col,'rois','subj_num'])
        if len(drop_cols) > 0:
            edges.drop(columns=(drop_cols),inplace=True)
        edges = edges.replace({np.nan:0})
        edges = edges.to_numpy() 
        edges = edges + edges.T - np.diag(np.diag(edges))
        edges_bin = np.where(edges>.1,1,0)

        group_name = ''.join(self.group) if self.group else 'sample'
        np.savetxt(os.path.join(shared.conn_dir, str(shared.date) + '_' + self.network + '_' + \
                   group_name + '_' + str(self.prop_thr).split('.')[-1] + '_bnv.edge'), edges, delimiter='\t')
        np.savetxt(os.path.join(shared.conn_dir, str(shared.date) + '_' + self.network + '_' + group_name + \
                   '_' + str(self.prop_thr).split('.')[-1] + '_binary_bnv.edge'), edges_bin, delimiter='\t')

    def make_pval_edge_file(self):
        edges,x = fam.get_sig_edges(self.network,prop_thr=self.prop_thr)
        edges = edges + edges.T - np.diag(np.diag(edges))
        edges_bin = np.where(edges>0,1,np.where(edges<0,-1,0))

        group_name = ''.join(self.group) if self.group else 'sample'

        np.savetxt(os.path.join(shared.conn_dir, str(shared.date) + '_' + self.network + '_' + \
                   group_name + '_' + str(self.prop_thr).split('.')[-1] + '_pval_bnv.edge'), edges, delimiter='\t')
        np.savetxt(os.path.join(shared.conn_dir, str(shared.date) + '_' + self.network + '_' + group_name + \
                   '_' + str(self.prop_thr).split('.')[-1] + '_binary_pval_bnv.edge'), edges_bin, delimiter='\t')

    def run_bnv_prep(self,statistical_edges=False):
        self.clean_labels()
        self.load_summary_data()
        self.make_node_file()
        if statistical_edges:
            self.make_pval_edge_file()
        else:
            self.make_edge_file()
