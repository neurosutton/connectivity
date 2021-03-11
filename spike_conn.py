import os
import os.path as op
from collections import Counter
from mne.io import read_raw_fif
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne import (make_forward_solution, read_source_spaces, read_cov, morph,
                 read_labels_from_annot, read_cov)
from mne.connectivity import spectral_connectivity
from mne.preprocessing import maxwell_filter, ICA
import numpy as np
import csv
import sys

mri_root_dir = '/Users/joshbear/clinical/meg/anat/'
subjects_root_dir = '/Users/joshbear/research/eses_connectivity/data/subjects'

# subject = input('Subject name: ')
# subject = 'ESES_01'

# raw_dir = op.join(subjects_root_dir, subject, 'meg', 'data')
# pipeline_dir = op.join(subjects_root_dir, subject, 'meg', 'pipeline')
# output_dir = op.join(subjects_root_dir, subject, 'meg', 'output')
# anat_dir = op.join(subjects_root_dir, subject, 'meg', 'anat')

# temporarily, will just default to the correct file
# in the future, though, need to make this more sophisticated
events_file = 'rest_1_clean-raw_events.txt'

# run = 'rest_1'
# fname_clean = op.join(raw_dir, run + '_clean-raw.fif')
# fname_empty_room = op.join(raw_dir, 'empty_room-raw.fif')
# fname_trans = op.join(raw_dir, run + '-trans.fif')
# fname_bem_sol = op.join(anat_dir, subject + '-bem-sol.fif')
# fname_src = op.join(anat_dir, subject + '-src.fif')
# fname_noise_cov = op.join(pipeline_dir, 'noise-cov.fif')

# events_in = open(op.join(pipeline_dir, events_file), 'r')
# events_lines = events_in.readlines()


def make_events_dict(file_path):
    events = []
    try:
        events_in = open(file_path, 'r')
    except FileNotFoundError:
        print('Tried to open ' + file_path + '\nNo such file. Exiting.')
        sys.exit()
    events_lines = events_in.readlines()
    for event in events_lines:
        type, start, stop = event.strip().split(',')
        event_dict = {}
        event_dict['type'] = type
        event_dict['start'] = float(start)
        event_dict['stop'] = float(stop)
        events.append(event_dict)
    events_in.close()
    return events


def print_events(events):
    print(Counter(event['type'] for event in events))


# events = make_events_dict(op.join(pipeline_dir, events_file))
# print_events(events)

"""Steps needed for processing the clean cropped files.

( 1) Create the cropped segment
( 1) Make forward solution
( 2) Make inverse source
( 3) Make morphed source
( 4) Generate connectivity matrix
( 5) Save the connectivity matrix
"""


def get_raw_segment(fname, start, stop):
    full_raw = read_raw_fif(fname)
    if (full_raw.n_times - 1) / full_raw.info['sfreq'] < stop:
        raise ValueError('Stop time for cropped segment exceeds end of original file.')
    raw = full_raw.copy().crop(start, stop)
    raw.load_data()
    # raw.resample(120)  # removed from the clinical scans so I can do higher frequencies
    print('*** get_raw_segment ***')
    print('*** Start: ' + str(start))
    print('*** Stop: ' + str(stop))
    return raw


def clean_raw_segment(raw, noise_cov_file=('/Users/joshbear/research/eses_connectivity/data/subjects/' +
                                                      'ESES_01/meg/pipeline/noise-cov.fif')):
    noise_cov = read_cov(noise_cov_file)
    # if bads is None:
        # bads = ['MEG 229', 'MEG 156', 'MEG 066']  # for E-0181
    # raw.info['bads'] = bads  # this is part of the 7/14/19 code..
    raw.pick_types(meg=True, ecg=True, exclude='bads')
    clean = maxwell_filter(raw, st_only=True,
                                             st_duration=1.0)
    """
    clean.filter(l_freq=12, h_freq=None)
    ica = ICA(n_components=0.95, method='fastica', random_state=0,
              noise_cov=noise_cov, max_iter=500)
    clean.pick_types(meg=True, ecg=True, exclude='bads')
    ica.fit(clean, decim=3,
            reject=None,
            verbose='warning')
    ecg_inds, ecg_scores = ica.find_bads_ecg(clean,
                                             ch_name='EEG 001',
                                             method='ctps')
    ica.exclude += ecg_inds
    print('Applying ICA.')
    ica.apply(clean)
    clean = clean.filter(l_freq=1, h_freq=70)
    """
    return clean


def get_forward_solution(raw, fname_src, fname_trans, fname_bem_sol):
    src = read_source_spaces(fname_src)
    fwd = make_forward_solution(raw.info,
                                fname_trans,
                                src,
                                fname_bem_sol,
                                mindist=5.0,
                                meg=True,
                                eeg=False,
                                n_jobs=1)
    return fwd


def get_inverse_operator(raw, fwd, fname_noise_cov):
    cov = read_cov(fname_noise_cov)
    inv_op = make_inverse_operator(raw.info,
                                   fwd,
                                   cov,
                                   depth=None,
                                   fixed=False)
    return inv_op


def get_morphed_source(stc, mri_root_dir, subject):
    vertices_to = [np.arange(10242), np.arange(10242)]
    stc_to = stc.morph(subject_to='fsaverage', grade=vertices_to, smooth=None,
                       subjects_dir=mri_root_dir, buffer_size=64, n_jobs=1,
                       subject_from=subject, sparse=False, verbose=None)
    # stc_to = morph(subject, grade=vertices_to, 'fsaverage', stc, n_jobs=1,
    #               subjects_dir=mri_root_dir)
    return stc_to


def get_connectivity_matrix(stc, sfreq, fmin=4, fmax=24, mode='mean_flip',
                            method='imcoh', parc='aparc.a2009s',
                            subjects_dir=mri_root_dir):
    fs_src = read_source_spaces('/Users/joshbear/research/epi_conn/' +
                                'fsaverage/anat/fsaverage-src.fif')
    labels_parc = read_labels_from_annot(
            'fsaverage', parc=parc,
            subjects_dir=subjects_dir)
    label_ts = stc.extract_label_time_course(labels_parc, fs_src,
                                             mode=mode,
                                             allow_empty=True)
    label_ts = [label_ts]
    con = spectral_connectivity(label_ts, method=method, mode='multitaper',
                                sfreq=sfreq, fmin=fmin, fmax=fmax,
                                faverage=True, mt_adaptive=True, n_jobs=1)
    return con[0][:, :, 0]


def get_volume_connectivity_matrix(stc, sfreq, fname_vol_src, fmin=4, fmax=24, mode='mean_flip',
                            method='imcoh',
                            subjects_dir=mri_root_dir):
    src = read_source_spaces(fname_vol_src)
    con = spectral_connectivity([stc.data], method=method, mode='multitaper',
                                sfreq=sfreq, fmin=fmin, fmax=fmax,
                                faverage=True, mt_adaptive=True, n_jobs=1)
    return con[0][:, :, 0]


def write_connectivity_matrix(con, fname, fill_matrix=True):
    if fill_matrix is True:
        for i in range(len(con[1])):
            for j in range(i, len(con[0])):
                con[i][j] = con[j][i]
    with open(fname, 'w+') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=' ')
        csvWriter.writerows(con)


def create_connectivity_matrix(method='dSPM'):
    snr = 1.0
    lambda2 = 1.0 / snr ** 2
    raw = get_raw_segment()
    fwd = get_forward_solution(raw)
    inv_op = get_inverse_operator(raw, fwd)
    stc = apply_inverse_raw(raw,  inv_op, lambda2, method)
    stc_to = get_morphed_source(stc)
    con = get_connectivity_matrix(stc_to, raw.info['sfreq'])
    return con
