import sys
sys.path.append('/Users/joshbear/research/eses_connectivity/scripts')
import analyze_conn_output as aco
import glob
import os.path as op
import numpy as np
import scipy.io
from scipy import stats
from itertools import compress
from importlib import reload

network_dmn_r = [132, 134, 135]
network_dmn_l = [132, 133, 135]
network_visual_r = [139, 140, 142]
network_visual_l = [139, 140, 141]
network_attention_r = [151, 153]
network_attention_l = [150, 152]

networks = {'dmn': [132, 133, 134, 135],
            'somatomotor': [136, 137, 138],
            'visual': [139, 140, 141, 142],
            'salience': [143, 144, 145, 146, 147, 148, 149],
            'salience_r': [143, 145, 147, 149],
            'salience_l': [143, 144, 146, 148],
            'attention': [150, 151, 152, 153],
            'frontoparietal': [154, 155, 156, 157],
            'frontoparietal_r': [155, 157],
            'frontoparietal_l': [154, 156],
            'language': [158, 159, 160, 161],
            'language_r': [159, 161],
            'language_l': [158, 160],
            'cerebellar': [162, 163]}

study_dir = '/Users/joshbear/research/eses_connectivity/'
analysis_dir = op.join(study_dir, 'analyses/conn_prelim/k23_prelim_32',
                       'results/firstlevel/SBC_01')
# example_filename ='resultsROI_Subject001_Condition001.mat'

results_files = glob.glob(op.join(analysis_dir, 'resultsROI_Subject*.mat'))
results_files.sort()

eses = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 1, 1, 1, 1, 1,
                 1, 1])
hc = np.array(1 - eses)

ieds_right = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
ieds_left = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
cects_focal = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
cects_bil = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

tmp = scipy.io.loadmat(results_files[0])
tmp['names'][0][154:158]

eses_files = list(compress(results_files, np.nan_to_num(eses)))
hc_files = list(compress(results_files, np.nan_to_num(hc)))
ied_r_files = list(compress(results_files, ieds_right))
ied_l_files = list(compress(results_files, ieds_left))
cects_focal_files = list(compress(results_files, cects_focal))
cects_bil_files = list(compress(results_files, cects_bil))

aco.test_network_differences(eses_files, hc_files, networks['language'])
aco.test_network_differences(eses_files, hc_files, networks['dmn'])
aco.test_network_differences(eses_files, hc_files, networks['frontoparietal'])
aco.test_network_differences(eses_files, hc_files, networks['attention'])
aco.test_network_differences(eses_files, hc_files, networks['visual'])
aco.test_network_differences(eses_files, hc_files, networks['somatomotor'])
aco.test_network_differences(eses_files, hc_files, networks['salience'])

aco.get_network_mean(eses_files, networks['salience'])
aco.get_network_mean(hc_files, networks['salience'])
aco.get_network_mean(ied_r_files, networks['salience'])
aco.get_network_mean(ied_l_files, networks['salience'])

aco.get_network_mean(eses_files, networks['salience_r'])
aco.get_network_mean(hc_files, networks['salience_r'])
aco.get_network_mean(ied_r_files, networks['salience_r'])
aco.get_network_mean(ied_l_files, networks['salience_r'])

aco.get_network_mean(eses_files, networks['salience_l'])
aco.get_network_mean(hc_files, networks['salience_l'])
aco.get_network_mean(ied_r_files, networks['salience_l'])
aco.get_network_mean(ied_l_files, networks['salience_l'])

aco.get_network_mean(cects_focal_files, networks['salience_l'])
aco.get_network_mean(cects_focal_files, networks['salience_r'])
aco.get_network_mean(cects_focal_files, networks['frontoparietal_l'])
aco.get_network_mean(cects_focal_files, networks['frontoparietal_r'])
aco.get_network_mean(cects_focal_files, networks['language_l'])
aco.get_network_mean(cects_focal_files, networks['language_r'])

aco.get_network_mean(cects_bil_files, networks['salience_l'])
aco.get_network_mean(cects_bil_files, networks['salience_r'])
aco.get_network_mean(cects_bil_files, networks['frontoparietal_l'])
aco.get_network_mean(cects_bil_files, networks['frontoparietal_r'])
aco.get_network_mean(cects_bil_files, networks['language_l'])
aco.get_network_mean(cects_bil_files, networks['language_r'])

aco.get_network_mean(hc_files, networks['language_l'])
aco.get_network_mean(hc_files, networks['language_r'])
aco.get_network_mean(eses_files, networks['language_l'])
aco.get_network_mean(eses_files, networks['language_r'])


language_eses = np.nanmean(aco.get_network_values_for_list(eses_files, networks['language']))
salience_eses = np.nanmean(aco.get_network_values_for_list(eses_files, networks['salience']))
frontoparietal_eses = np.nanmean(aco.get_network_values_for_list(eses_files, networks['frontoparietal']))
language_hc = np.nanmean(aco.get_network_values_for_list(hc_files, networks['language']))
salience_hc = np.nanmean(aco.get_network_values_for_list(hc_files, networks['salience']))
frontoparietal_hc = np.nanmean(aco.get_network_values_for_list(hc_files, networks['frontoparietal']))

aco.plot_network_differences([language_eses, salience_eses, frontoparietal_eses],
                             [language_hc, salience_hc, frontoparietal_hc],
                             ['Language', 'Salience', 'Frontoparietal'])
