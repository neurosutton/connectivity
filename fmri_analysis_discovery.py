"""
Explore connectivity matrices generated through preprocessing in CONN, using the Glasser atlas.
Date: August 2020
@author: Josh Bear, MD
"""

import fmri_analysis_functions as faf
from scipy.io import loadmat
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from importlib import reload
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr

conn_dir = ''
data_dir = op.join(conn_dir,'conn_project01_results')
file = 'resultsROI_Condition001.mat'
subject_file =  op.join(conn_dir,'conn_project01_results/eses_subjects_202008.csv')

mdata = loadmat(op.join(data_dir, file))
subj_data = pd.read_csv(subject_file)
eses_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'eses']
hc_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'hc']


subj_data
faf.plot_cohort_comparison(mdata, '', eses_indices, hc_indices, vmin=-0.6, vmax=0.6)

subj_data.head()

faf.plot_cohort_comparison(mdata, 'Language_R', eses_indices, hc_indices, vmin=-0.6, vmax=0.6)

faf.plot_cohort_comparison(mdata, '', eses_indices, hc_indices, vmin=-0.6, vmax=0.6)

hc_indices = [i for i, x in enumerate(list(subj_data['group'])) if x == 'hc']

lang_subjects = []
lang_scores = []
global_scores = []
dmn_conn = []
lang_conn = []
brain_conn = []

for subj, sc in enumerate(list(subj_data['np_verbal'])):
    if not np.isnan(sc):
        lang_subjects.append(subj)
        lang_scores.append(sc)

lang_conn = [x for x in subj_data['np_global']]

lang_subjects
lang_scores

for subj in lang_subjects:
    m = faf.get_network_matrix(mdata, 'Language_L', subj)
    m[np.triu_indices(m.shape[0], k=0)] = np.nan
    lang_conn.append(np.nanmean(m))
    md = faf.get_network_matrix(mdata, 'Default', subj)
    md[np.triu_indices(md.shape[0], k=0)] = np.nan
    dmn_conn.append(np.nanmean(md))
    mb = faf.get_network_matrix(mdata, '', subj)
    mb[np.triu_indices(md.shape[0], k=0)] = np.nan
    brain_conn.append(np.nanmean(md))

plt.scatter(lang_conn, lang_scores)

reload(faf)
faf.plot_score_by_network(subject_file, 'np_verbal', mdata, 'Language_L')

faf.plot_score_by_network(subject_file, 'np_exec', mdata, 'Fronto')

faf.plot_score_by_network(subject_file, 'np_global', mdata, 'Default')



scores = faf.get_subject_scores(subject_file, 'np_exec')

# index #28 (eses_29) has a thalamic stroke, which seems to have led to massive
# increase in connectivity compared to others; let's remove from discovery
# others with abnormal mri: #25 (PMG), #27 (insular)
scores.pop(28, None)
scores.pop(25, None)
scores.pop(27, None)

score_values = []
conn_values = []
for subj in scores.keys():
    m = faf.get_network_matrix(mdata, 'Fronto', subj)
    m[np.triu_indices(m.shape[0], k=0)] = np.nan
    m = np.nanmean(m)
    conn_values.append(m)
    score_values.append(scores[subj])

plt.scatter(conn_values, score_values)
scores

conn_values
np.corrcoef(conn_values, score_values)
pearsonr(conn_values, score_values)

seaborn.regplot(conn_values, score_values)


# quick check to show network-specific strenght by subject..
reload(faf)
faf.plot_network_strength_by_subject(subject_file, mdata, 'Default', abs=False, sign='positive')

faf.describe_cohort_networks(mdata, '', eses_indices, hc_indices, name_1='ESES', name_2='HC')

faf.describe_cohort_networks(mdata, 'Default', eses_indices, hc_indices, name_1='ESES', name_2='HC', comparison='all', sign='positive')
ms = faf.get_cohort_network_matrices(mdata, 'Default', eses_indices, mean='subject')
ms.shape
ms

faf.get_cohort_network_matrices(mdata, 'Default', eses_indices, mean='subject', sign='positive')

eses_indices
eses_nmri = [0, 1, 3, 4, 19, 23, 24, 26, 29]

faf.describe_cohort_networks(mdata, 'Cingulo', eses_nmri, hc_indices, name_1='ESES', name_2='HC', comparison='subject', sign='positive')



# Let's look at some more of the network/function relationships.
scores = faf.get_subject_scores(subject_file, 'np_global')

# index #28 (eses_29) has a thalamic stroke, which seems to have led to massive
# increase in connectivity compared to others; let's remove from discovery
# others with abnormal mri: #25 (PMG), #27 (insular)
scores.pop(28, None)
# scores.pop(25, None)
# scores.pop(27, None)

score_values = []
conn_values = []
for subj in scores.keys():
    m = faf.get_network_matrix(mdata, 'Default', subj)
    m[np.triu_indices(m.shape[0], k=0)] = np.nan
    m = np.nanmean(m)
    conn_values.append(m)
    score_values.append(scores[subj])

pearsonr(conn_values, score_values)
seaborn.regplot(conn_values, score_values)

# LANGUAGE_L
scores = faf.get_subject_scores(subject_file, 'np_verbal')

# index #28 (eses_29) has a thalamic stroke, which seems to have led to massive
# increase in connectivity compared to others; let's remove from discovery
# others with abnormal mri: #25 (PMG), #27 (insular)
scores.pop(28, None)
scores.pop(25, None)
scores.pop(27, None)

score_values = []
conn_values = []
for subj in scores.keys():
    m = faf.get_network_matrix(mdata, 'Language_L', subj)
    m[np.triu_indices(m.shape[0], k=0)] = np.nan
    m = np.nanmean(m)
    conn_values.append(m)
    score_values.append(scores[subj])

pearsonr(conn_values, score_values)
seaborn.regplot(conn_values, score_values)


# DEFAULT MODE AND WORKING MEMORY
scores = faf.get_subject_scores(subject_file, 'np_workingmem')

# index #28 (eses_29) has a thalamic stroke, which seems to have led to massive
# increase in connectivity compared to others; let's remove from discovery
# others with abnormal mri: #25 (PMG), #27 (insular)
scores.pop(28, None)
scores.pop(25, None)
scores.pop(27, None)

score_values = []
conn_values = []
for subj in scores.keys():
    m = faf.get_network_matrix(mdata, 'Default', subj)
    m[np.triu_indices(m.shape[0], k=0)] = np.nan
    m = np.nanmean(m)
    conn_values.append(m)
    score_values.append(scores[subj])

pearsonr(conn_values, score_values)
seaborn.regplot(conn_values, score_values)

# Executive Function and Frontoparietal Network
scores = faf.get_subject_scores(subject_file, 'np_exec')

# index #28 (eses_29) has a thalamic stroke, which seems to have led to massive
# increase in connectivity compared to others; let's remove from discovery
# others with abnormal mri: #25 (PMG), #27 (insular)
scores.pop(28, None)
scores.pop(25, None)
scores.pop(27, None)

score_values = []
conn_values = []
for subj in scores.keys():
    m = faf.get_network_matrix(mdata, 'Fronto', subj)
    m[np.triu_indices(m.shape[0], k=0)] = np.nan
    m = np.nanmean(m)
    conn_values.append(m)
    score_values.append(scores[subj])

pearsonr(conn_values, score_values)
seaborn.regplot(conn_values, score_values)
