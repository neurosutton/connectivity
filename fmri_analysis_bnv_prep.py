"""
Create files for BrainNetViewer (BNV) visualization of network connectivity.

Date: December 2020
@author: Brianne Sutton, PhD
v0.1
"""
import pandas as pd
import os
import fmri_analysis_utilities as utils
import fmri_analysis_manipulations as fam
import fmri_analysis_load_funcs as faload

shared = faload.load_shared()

class bnv_analysis():
    def __init__(self, network=None, label_file=os.path.join(shared.atlas_dir,'hcpmmp1_expanded_labels.csv'), group=None, atlas_label="SuttonLabel", subject_list=None, prop_thr=0.7):
        self.network = network
        self.group = group #Filtered group for the node values
        self.atlas_label = atlas_label
        self.label_df = pd.DataFrame(pd.read_csv(label_file))
        self.subject_list = subject_list
        self.prop_thr = prop_thr

    def clean_labels(self):
        """Reduce mismatches and extraneous information from label file, so that the bare minimum needed for BNV is merged."""
        self.label_df = self.label_df[['x','y','z',self.atlas_label]]

    def load_data(self,analyze=[]):
        """Allows string input for group comparison"""
        if not analyze:
            grp_dict = {'shared.group1':shared.group1,'shared.group2':shared.group2}
            compare = 'no'
        else:
            grp_dict = {k:v for k,v in shared.__dict__.items() if any(x in analyze for x in v)}
            compare = 'yes'

        network_mask = fam.make_proportional_threshold_mask(shared.network, shared.prop_thr, exclude_negatives=exclude_negatives)
        dfs = []
        for k in grp_dict.keys():
            indices = shared.__dict__[k.split('.')[-1]+'_indices'] # Flexible solve for group 1 or 2, depending on the group id from analyze
            df = pd.DataFrame(get_cohort_network_matrices(shared.network, indices, mean=False, conn_data=None, prop_thr=shared.prop_thr, subject_level=False, network_mask=network_mask, exclude_negatives=False))
            df['group'] = grp_dict[k]
            dfs.append(df)
        df = pd.concat(dfs)
        if compare == 'yes':
            groups = set(df['group'])
            df1 = df.loc[df['group']==groups[0],:].drop(subset=['group'])
            df2 = df.loc[df['group']==groups[1],:].drop(subset=['group'])
            df =  df1-df2 
            df['group'] = 'diff'
        parcels = get.get_network_parcels(network_name)
        df['rois'] = list(parcels.keys())
        return df

    #WHAT does load_data output. Is it one value per row for each ROI???
    def make_node_file(self):

        if self.check == 'done':      
            node_df = self.conn_df[[msr_of_int] + ['rois']]
            #node_df[msr_of_int] = node_df[msr_of_int].mask(~node_df['rois'].str.lower().str.contains(self.network)) # This seems to be a redundant filter for network ROIs. Might be okay, but need to test
        
            out_df = label_df.merge(node_df, left_on = self.atlas_label, right_on = 'rois').drop(columns=([self.atlas_label]))
            out_df['size'] = out_df[msr_of_int]
            out_df = out_df.apply(lambda x: x.str.strip() if x.dtype=="object" else x)
            out_df = out_df[['x','y','z',msr_of_int,'size','rois']]
            out_df.drop_duplicates(inplace=True)
            out_df.to_csv(op.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + self.prop_thr + '_bnv.node'),header=False, index=False,sep='\t')
        else:
            print(f'Please validate that you are using the correct data by running {self.clean_data}')

    def make_edge_file(self, binary=True):        
        parcels = get.get_network_parcels(self.network, mdata=shared.mdata)
        indices = list(parcels.values())
        edges = self.conn_df
        edges.drop(columns=([shared.name_id_col, shared.group_id_col]),inplace=True)
        edges = edges.replace({np.nan:0})
        edges_bin = np.where(edges>.1,1,0)
        np.savetxt(op.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + self.prop_thr + '_bnv.edge'),edges,delimiter='\t')

    def run_bnv_prep(self):
        self.clean_labels()
        self.load_data()
        self.clean_data()
        self.make_node_file()
        self.make_edge_file()
