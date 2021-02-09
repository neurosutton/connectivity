"""
Load data from CONN output and create shared variables for other modules in package.

Date: December 2020
@author: Brianne Sutton, PhD from Josh Bear, MD
v0.3 (BMS) Adapted fmri_analysis_functions for pandas
"""

import os,sys
from glob import glob
import numpy as np
import fnmatch, random, time, pickle
import pandas as pd
import json
from collections import OrderedDict, defaultdict
from datetime import datetime
from scipy.io import loadmat



class init_variables():
    """Provides a single mechanism to pass shared variables around.
    Initial calls to the method should supply the filepath to the definitions file with all the group info, filepaths to directories, etc. in it."""
    def __init__(self,__def_path__=None):
        if __def_path__:
            self.__def_path__ = __def_path__
        else:
            self.__def_path__ = os.path.dirname(__file__)
        self.date = datetime.today().strftime('%Y%m')
        with open(os.path.join(self.__def_path__,'directory_defs.json')) as f:
            defs = json.load(f)
            self.conn_dir = defs['conn_dir']
            self.main_dir = defs['main_dir']
            self.atlas_dir = defs['atlas_dir']
            self.proj_dir = defs['data_dir']
            global pkl_file
            #pkl_file =  os.path.join(self.proj_dir,'vars.pkl')
            pkl_file =  os.path.join(self.__def_path__,'shared.py')
            self.pkl_file = pkl_file
            self.name_id_col = defs['name_id_col']
            self.group_id_col = defs['group_id_col']
            self.group1 = defs['group1']
            self.group2 = defs['group2']
        self.nonimaging_subjectlevel_data =  os.path.join(self.main_dir,'eses_subjects_202008.csv')
        self.conn_file = 'resultsROI_Condition001.mat'
        self.excl_negatives = False
        _pickle(obj=self)



def _pickle(obj):
    if not 'pkl_file' in globals():
        global pkl_file
        pkl_file = init_variables().pkl_file
    with open(pkl_file,'w+') as f:
 #       pickle.dump(obj,f)
        for k,v in obj.__dict__.items():
            if ("__") in k :
                pass
            elif k in ['mdata','conn_data']:
                pass
            elif ("[") in str(v):
                f.write(f"{k} = {v}\n")
            else:
                f.write(f"{k} = '{v}'\n")

def load_shared(pkl_file):
    try:
        with open(pkl_file, 'rb') as f:
            tmp = pickle.load(f)
        if hasattr(tmp,'conn_data'):
            return tmp.mdata, tmp.conn_data
    except NameError:
        print('Error: Please instantiate shared_variables (faload) before importing other modules.')

def update_shared(obj=None):
    if obj:
        _pickle(obj)
#    global shared
#    shared = load_shared()
#    return shared

if __name__ == "__main__":
    x = init_variables()
