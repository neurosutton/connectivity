from scipy.io import loadmat,savemat
import pandas as pd
import numpy as np
from glob import glob
import os, shutil
import nibabel as nb
from nipype.interfaces.fsl import Merge

class process_scans():
    def __init__(self,nii, art_file, home_dir):
        self.nii = nii
        self.home_dir = home_dir
        self.art_file = art_file
        self.rest_dir,x = os.path.split(self.art_file)
        os.chdir(self.rest_dir)

    # fslsplit
    def fslsplit(self):
        temp,x = os.path.split(self.nii)
        print('Splitting {}'.format(self.nii))
        !fslsplit $self.nii -t
        self.split_vols = sorted(glob(os.path.join(self.rest_dir,'vol*')))
        #split_vols = glob(os.path.join(self.home_dir,'vol*')) # Why FSL insists on putting this here, I don't know.

    # counter loop through art_regression_outliers until collected 246 volumes
    def find_vols(self, vols=246):

        mat = loadmat(self.art_file)
        mdata = mat['R']
        min_extra_vols = mdata[:vols,:].sum()
        if min_extra_vols == 0 or not min_extra_vols:
            print('{} has NO bad volumes'.format(self.rest_dir))
        else:
            print('{} volumes available.\n{} volumes of bad data in those first files.'.format(len(mdata),min_extra_vols))
            while min_extra_vols > 0 and min_extra_vols <= vols:
                end_vol = int(vols + min_extra_vols)
                min_extra_vols = mdata[vols:end_vol,:].sum()
                print('Next eligible chunk had {} outliers.'.format(min_extra_vols))
                vols = int(end_vol + min_extra_vols) # update to include the current last volume, so that another iteration can pickup the outlier search where this loop leaves off

        print('Checking if the number of volumes ({}) is greater than needed ({})'.format(len(self.split_vols),vols))
        if len(self.split_vols) >= vols:
            extras_dir = self.rest_dir + '_extraVols'
            print('Creating extras directory, {}, and moving {} files to it.'.format(extras_dir,len(mdata)-vols))
            if not os.path.isdir(extras_dir):
                os.makedirs(extras_dir)

            if not self.split_vols:
                # should not be stored here, but FSL is stubborn
                tmp,y = os.path.split(self.rest_dir)
                self.split_vols = sorted(glob(os.path.join(tmp,'vol*')))
            for vol in self.split_vols:
                vol_filename = os.path.basename(vol)
                v = vol_filename[3:7] # get the volume number
                if int(v) > vols: # don't make it = to vols, b/c want to be volume number inclusive
                    try:
                        shutil.move(vol, os.path.join(extras_dir,vol_filename))
                    except Exception as e:
                        print(e)
                    if os.path.isfile(vol):
                        os.remove(vol)
        self.vols = vols
        print('Using {} volumes to keep consistent averaging.'.format(vols))

    def truncate_timeseries(self):
        art_files = glob(os.path.join(self.rest_dir,'art_reg*'))
        in_files = glob(os.path.join(self.rest_dir,'rp_e*'))
        # replace the mat file with the individually truncated one
        for f in in_files:
            print('Changing {}'.format(f))
            df = pd.DataFrame(pd.read_csv(f, header=None))
            df.replace('"',inplace=True)
            df = df.iloc[:self.vols,:]
            df.to_csv(f,index=False,sep='\t',header=False)
        for a in art_files:
            print('Changing {}'.format(a))
            mat = loadmat(a)  # load mat-file
            mdata = mat['R']  # variable in mat file
            mat['R'] = mdata[0:self.vols,:]
            savemat(a,mat)
