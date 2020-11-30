"""
Create files for BrainNetViewer (BNV) visualization of network connectivity.

Date: December 2020
@author: Brianne Sutton, PhD
v0.1
"""
import pandas as pd
import os
import fmri_analysis_functions as faf
import fmri_analysis_matrices as fam
import fmri_analysis_load_funcs as faload

name_id_col, group_id_col, data_dir, conn_dir = faload.study_logistics()

class analysis():
    def __init__(self, network, label_file=os.path.join(data_dir,'masks/hcpmmp1_expanded_labels.csv'), groups=['hc','eses'], atlas_label="SuttonLabel"):
        self.network = network
        self.groups = groups
        self.atlas_label = atlas_label
        self.label_df = pd.DataFrame(pd.read_csv(label_file))

    def clean_labels(self):
        """Reduce mismatches and extraneous information from label file, so that the bare minimum needed for BNV is merged."""
        self.label_df = self.label_df[['x','y','z',self.atlas_label]]

    def load_data(data_file):
        """Input from functional connectivity matrices that have been manipulated to include data for specific networks and thresholds."""
        

    def clean_data():
        """Remove unnecessary columns and insure that names will be consistent with merging parameters."""
        pass

    def make_node_file(df, coi, noi):
        coi is column of interest; noi is network of interest
        df = df[[coi] + ['rois']]
        df[coi] = df[coi].mask(~df['rois'].str.lower().str.contains(noi))
        
        out_df = label_df.merge(df, left_on = self.atlas_label, right_on = 'rois').drop(columns=([self.atlas_label]))
        out_df['size'] = out_df[coi]
        out_df = out_df.apply(lambda x: x.str.strip() if x.dtype=="object" else x)
        out_df = out_df[['x','y','z',coi,'size','rois']]
        out_df.drop_duplicates(inplace=True)
        out_df.to_csv(op.join(conn_dir,'test_bnv.node'),header=False, index=False,sep='\t')

    def make_edge_file():
        dat = fam.analysis(mdata, network_name='fronto', subj_list='eses020')
        edges = dat.create_matrix()
        edges.drop(columns=(['BK_name','group']),inplace=True)
        edges = edges.replace({np.nan:0})
        edges_bin = np.where(edges>.1,1,0)
        np.savetxt(op.join(conn_dir,'test_bnv.edge'),edges,delimiter='\t')