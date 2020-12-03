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
    def __init__(self, network=None, label_file=os.path.join(shared.atlas_dir,'hcpmmp1_expanded_labels.csv'), group=None, atlas_label="SuttonLabel", subject_list=None, prop_thr=0):
        self.network = network
        self.group = group #Filtered group for the node values
        self.atlas_label = atlas_label
        self.label_df = pd.DataFrame(pd.read_csv(label_file))
        self.subject_list = subject_list
        self.prop_thr = prop_thr

    def clean_labels(self):
        """Reduce mismatches and extraneous information from label file, so that the bare minimum needed for BNV is merged."""
        self.label_df = self.label_df[['x','y','z',self.atlas_label]]

    def load_data(self,data_file=None):
        """Input from functional connectivity matrices that have been limited to the network of interest."""
        if data_file:
            try:
                df = pd.DataFrame(pd.read_csv(data_file))
            except:
                try:
                    df = pd.DataFrame(pd.read_excel(data_file))
                except Exception as e:
                    print(e)
        else:
            dfs = []
            for df in shared.conn_data.shape[2]:


    def clean_data(self):
        """Manipulate df to include data for specific networks and thresholds.Remove unnecessary columns and insure that names will be consistent with merging parameters."""
        if self.subject_list:
            self.conn_df = utils.filter_conn_df_subjects(self.conn_df,subject_list)
        if self.group:
            self.conn_df = self.conn_df.loc[self.conn_df[shared.group_id_col]==self.group]
        self.conn_df = self.conn_df.loc[self.conn_df['prop_thr']==prop_thr,:]
        self.check = 'done'

    def make_node_file(self, msr_of_int):
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

    def make_edge_file(self):        
        if self.check == 'done':
            edges = self.conn_df
            edges.drop(columns=([shared.name_id_col, shared.group_id_col]),inplace=True)
            edges = edges.replace({np.nan:0})
            edges_bin = np.where(edges>.1,1,0)
            np.savetxt(op.join(shared.conn_dir,str(shared.date)+ '_' + self.network + '_' + self.prop_thr + '_bnv.edge'),edges,delimiter='\t')
        else:
            print(f'Please validate that you are using the correct data by running {self.clean_data}')

    def run_bnv_prep(self):
        self.clean_labels()
        self.load_data()
        self.clean_data()
        self.make_node_file()
        self.make_edge_file()
