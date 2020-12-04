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
    def __init__(self, network=None, label_file=os.path.join(shared.atlas_dir,'hcpmmp1_expanded_labels.csv'), group=None, atlas_label="SuttonLabel", subject_list=None, prop_thr=0.7, exclude_negatives=shared.excl_negatives):
        self.network = "wb" if network is None else network
        self.group = group #Filtered group for the node values
        self.atlas_label = atlas_label
        self.label_df = pd.DataFrame(pd.read_csv(label_file))
        self.clean_labels()
        self.subject_list = subject_list
        self.prop_thr = prop_thr
        self.exclude_negatives = exclude_negatives


    def clean_labels(self):
        """Reduce mismatches and extraneous information from label file, so that the bare minimum needed for BNV is merged."""
        self.label_df = self.label_df[['x','y','z',self.atlas_label]]
        self.label_df[self.atlas_label] = self.label_df[self.atlas_label].str.lower()
   
    def load_summary_data(self,analyze=None, bnv_node=True):
        """Allows string input for group comparison"""
        compare='no' # Tag for whether to subtract mean df's
        analyze = self.group if analyze is None else analyze

        if not analyze:
            grp_dict = {'shared.group1':shared.group1,'shared.group2':shared.group2}
        else:
            print('Looking for matching groups')
            if not isinstance(analyze,list):
                analyze = [analyze]
            if len(analyze) > 1:
                compare='yes' # more than one group
            grp_dict = {k:v for k,v in shared.__dict__.items() if str(v) in analyze}

        dfs = []
        for k in grp_dict.keys():
            indices = shared.__dict__[k.split('.')[-1]+'_indices'] # Flexible solve for group 1 or 2, depending on the group id from analyze
            print(indices)
            df = self.get_cohort_bnv_data(indices = indices, mean=bnv_node)
            df['group'] = grp_dict[k]
            dfs.append(df)
        df = pd.concat(dfs)
        df = df.reset_index().rename(columns={'index':'rois'})
        if compare == 'yes':
            df = self.calc_diff_df(df.drop(columns=['subj_num']))
        else:
            df = df.groupby('rois').mean().reset_index().rename(columns={'index':'rois'})
        return df


    def get_cohort_bnv_data(self, indices=[1], mean=False):
        network_mask = fam.make_proportional_threshold_mask(self.network, self.prop_thr, exclude_negatives=self.exclude_negatives)
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
            else:
                s = pd.DataFrame(s.groupby('rois').mean(),columns=['fc'])
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

    def make_node_file(self, msr_of_int='fc', analyze=None):
        analyze = self.group if analyze is None else analyze
        fc_df = self.load_summary_data(analyze=analyze, bnv_node=True)
        node_df = fc_df[[msr_of_int] + ['rois']]
        out_df = self.label_df.merge(node_df, left_on = self.atlas_label, right_on = 'rois').drop(columns=([self.atlas_label]))
        out_df['size'] = out_df[msr_of_int]
        out_df = out_df.apply(lambda x: x.str.strip() if x.dtype=="object" else x)
        out_df = out_df[['x','y','z',msr_of_int,'size','rois']]
        out_df.drop_duplicates(inplace=True)
        out_df = out_df.replace({np.nan:0})
        out_df.to_csv(os.path.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + ''.join(self.group) + '_' + str(self.prop_thr).split('.')[-1] + '_bnv.node'), header=False, index=False,sep='\t')


    def make_edge_file(self, analyze=None, mdata=None): 
        analyze = self.group if analyze is None else analyze
        mdata = get.get_mdata() if mdata is None else mdata
        parcels = get.get_network_parcels(self.network, mdata=mdata)
        indices = list(parcels.values())
        edges = self.load_summary_data(analyze=analyze, bnv_node=False)
        drop_cols = edges.columns.intersection([shared.name_id_col, shared.group_id_col,'rois','subj_num'])
        if len(drop_cols) > 0:
            edges.drop(columns=(drop_cols),inplace=True)
        edges = edges.replace({np.nan:0})
        edges = edges.to_numpy() 
        edges = edges + edges.T - np.diag(np.diag(edges))
        edges_bin = np.where(edges>.1,1,0)
        np.savetxt(os.path.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + ''.join(self.group) + '_' + str(self.prop_thr).split('.')[-1]  + '_bnv.edge'),edges,delimiter='\t')
        np.savetxt(os.path.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + ''.join(self.group) + '_'+ str(self.prop_thr).split('.')[-1]  + '_binary_bnv.edge'),edges_bin,delimiter='\t')

    def run_bnv_prep(self):
        self.clean_labels()
        self.load_summary_data()
        self.make_node_file()
        self.make_edge_file()
