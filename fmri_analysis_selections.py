import pandas as pd

def get_init_params():
  """Load study-level parameters for basic information"""

def get_group_membership(subj_df, group):
    if debug:
        print(subj_df.loc[subj_df[group_id_col]==group,group_id_col])
    else:
        pass
    return subj_df.index[subj_df[group_id_col]==group].tolist()

def get_group_membership_by_name(subj_df, group):
    return subj_df[name_id_col].loc[:,subj_df[group_id_col] == group]
