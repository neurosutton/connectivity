from mne import (Epochs, read_cov, read_forward_solution, read_source_spaces, compute_source_morph,
                 read_labels_from_annot, compute_source_morph)
from mne.connectivity import spectral_connectivity
from mne.io import read_raw_fif
from collections import Counter
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import os.path as op
import csv
from os import mkdir
import glob

"""
import os


from mne.io import read_raw_fif
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne import (make_forward_solution, read_source_spaces, read_cov, morph,
                 read_labels_from_annot, read_cov)

from mne.connectivity import spectral_connectivity
from mne.preprocessing import maxwell_filter, ICA

import sys



TO DO: Improve handling and display when epochs called with multiple possible event_ids.
TO DO: Make sure that 'bads' are identified from the Excel list and applied.
"""

def get_events_from_csv(file, sfreq=1000):
    """Create an events list from a BrainStorm events list."""
    events = []
    event_dict = {}
    try:
        events_in = open(file, 'r')
    except FileNotFoundError:
        print('Failed to open the given events file: ' + file)
        return None
    events_lines = events_in.readlines()
    for event in events_lines:
        name, seconds, *_ = event.strip().split(',')
        seconds = float(seconds)
        sample = int(seconds * sfreq)
        if name not in event_dict.keys():
            event_dict[name] = len(event_dict) + 1
        id = event_dict[name]
        events.append([sample, 0, id])
    return event_dict, events


def get_event_files(pipeline_dir):
    """Find the raw imported runs for given subject."""
    all_runs = []
    for file in glob.iglob(pipeline_dir + '/*.txt', recursive=False):
        all_runs.append(file)
    return sorted(all_runs)


def get_clean_raw_files(data_dir):
    all_files = []
    for file in glob.iglob(data_dir + '/clean*.fif', recursive=False):
        all_files.append(file)
    return sorted(all_files)


def get_multifile_events_dict(files, sfreq=1000):
    events = []
    event_dict = {}
    for file in files:
        try:
            events_in = open(file, 'r')
        except FileNotFoundError:
            print('Failed to open the given events file: ' + file)
            return None
        events_lines = events_in.readlines()
        for event in events_lines:
            name, seconds, *_ = event.strip().split(',')
            seconds = float(seconds)
            sample = int(seconds * sfreq)
            if name not in event_dict.keys():
                event_dict[name] = len(event_dict) + 1
            id = event_dict[name]
            event_data = [sample, 0, id]
            event_count = len(events)
            events.append(dict(count=event_count, data=event_data, file=file))
    return event_dict, events


def get_epochs(raw_file, events_file, t_start=0., t_window=1., event_id=None):
    raw = read_raw_fif(raw_file)
    event_dict, events = get_events_from_csv(events_file, raw.info['sfreq'])
    if event_id is None and len(event_dict) > 1:
        print('The following count of different events was identified:')
        print_events(events)
        raise ValueError('Event file contains multiple possible events, but no event_id was given.')
    tmin = t_start
    tmax = t_start + t_window
    epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)
    return epochs


def print_events(events):
    print(Counter(event[2] for event in events))


def get_pathnames_for_connectivity_analysis(subject, scan_type='eses'):
    """Return a dict containing all necessary file names to calculate connectivity."""
    mri_root_dir = '/Users/joshbear/research/eses_connectivity/fs_anatomy'
    subject_root_dir = op.join('/Users/joshbear/research/eses_connectivity/data', subject)
    src_file = op.join(subject_root_dir, 'meg', 'anat', subject + '-src.fif')
    if scan_type == 'eses':
        fwd_file = op.join(subject_root_dir, 'meg', 'pipeline', 'rest_1-fwd.fif')
        noise_file = op.join(subject_root_dir, 'meg', 'pipeline', 'noise-cov.fif')
        events_files = [op.join(subject_root_dir, 'meg', 'pipeline', 'rest_1-raw.txt'),
                        op.join(subject_root_dir, 'meg', 'pipeline', 'rest_2-raw.txt')]
        raw_files = [op.join(subject_root_dir, 'meg', 'data', 'rest_1_clean-raw.fif'),
                     op.join(subject_root_dir, 'meg', 'data', 'rest_2_clean-raw.fif')]
    elif scan_type == 'clinical':
        fwd_file = op.join(subject_root_dir, 'meg', 'pipeline', 'clean_rest_00-fwd.fif')
        noise_file = '/Users/joshbear/research/ied_network/data/subjects/ESES_01/meg/pipeline/noise-cov.fif'
        events_files = get_event_files(op.join(subject_root_dir, 'meg', 'pipeline'))
        raw_files = get_clean_raw_files(op.join(subject_root_dir, 'meg', 'data'))
    events_file = op.join(subject_root_dir, 'meg', 'data', 'rest_1-raw.txt')
    raw_file = op.join(subject_root_dir, 'meg', 'data', 'rest_1_clean-raw.fif')
    fs_src_file = '/Users/joshbear/research/epi_conn/fsaverage/anat/fsaverage-src.fif'
    output_dir = op.join(subject_root_dir, 'meg', 'output')
    return dict(mri_root_dir=mri_root_dir, subject_root_dir=subject_root_dir,
                src_file=src_file, fwd_file=fwd_file, noise_file=noise_file,
                events_file=events_file, raw_file=raw_file, fs_src_file=fs_src_file,
                output_dir=output_dir, events_files=events_files, raw_files=raw_files)


def write_connectivity_matrix(con, fname, fill_matrix=True):
    if fill_matrix is True:
        for i in range(len(con[1])):
            for j in range(i, len(con[0])):
                con[i][j] = con[j][i]
    with open(fname, 'w+') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=' ')
        csvWriter.writerows(con)


def run_eses_spike_conn_time_series(subject, time_start, time_stop, freqs=[[12, 30]],
                                    time_window=0.5, time_step=None, event=None, scan_type='eses'):
    """Calculate connectivity time series and save data for a list of frequencies and times."""
    pathnames = get_pathnames_for_connectivity_analysis(subject, scan_type=scan_type)
    noise_cov = read_cov(pathnames['noise_file'])
    fwd = read_forward_solution(pathnames['fwd_file'])
    labels_parc = read_labels_from_annot('fsaverage', parc='HCPMMP1',
                                         subjects_dir=pathnames['mri_root_dir'])
    fs_src = read_source_spaces(pathnames['fs_src_file'])

    if time_step is None:
        time_step = time_window  # defaults to no overlapping segments
    time_indices = np.arange(time_start, time_stop - time_window + time_step, time_step)
    snr = 1.0
    lambda2 = 1.0 / snr ** 2
    method = 'dSPM'
    m = []

    # events_dict, events = get_events_from_csv(pathnames['events_file'])
    events_dict, events2 = get_multifile_events_dict(pathnames['events_files'])

    if event is None:
        event_name = next(iter(events_dict))
        event_id = events_dict[event_name]
    elif type(event) is int:
        event_id = event
        event_name = list(events_dict.keys())[list(events_dict.values()).index(event)]
    elif type(event) is str:
        event_id = events_dict[event]
        event_name = event
    conn_output_dir = op.join(pathnames['output_dir'], str(event_name) + '_con_time_' + str(time_start) +
                              '-' + str(time_stop) + '_window_' + str(time_window) +
                              '_by_' + str(time_step))
    print(conn_output_dir)
    if op.isdir(conn_output_dir) is False:
        print('Making folder for ' + event_name + ' conn output.')
        mkdir(conn_output_dir)
    for fmin, fmax in freqs:
        band_output_dir = op.join(conn_output_dir, str(int(fmin)) + '-' + str(int(fmax)))
        if op.isdir(band_output_dir) is False:
            print('Making folder for ' + event_name + ' frequencies: ' + str(fmin) + '-' + str(fmax))
            mkdir(band_output_dir)
    print(band_output_dir)

    for fmin, fmax in freqs:
        for idx_time, time in enumerate(time_indices):
            print(f'Time: {time}, band: {fmin}-{fmax}')
            band_output_dir = op.join(conn_output_dir, str(int(fmin)) + '-' + str(int(fmax)))
            print(time)
            print(time_window)
            # epochs = get_epochs(pathnames['raw_file'], pathnames['events_file'], t_start=time,
            #                     t_window=time_window, event_id=event_id)
            # *** The following lines are attempting to integrate the possibility of clinical scans. ***
            all_epochs = []
            for count in range(len(pathnames['events_files'])):
                all_epochs.append(get_epochs(pathnames['raw_files'][count], pathnames['events_files'][count],
                                             t_start=time, t_window=time_window, event_id=event_id))
            # all_epochs = [get_epochs(pathnames['raw_files'][0], pathnames['events_files'][0], t_start=time,
            #                     t_window=time_window, event_id=event_id),
            #              get_epochs(pathnames['raw_files'][1], pathnames['events_files'][1], t_start=time,
            #                         t_window=time_window, event_id=event_id)]
            for epoch_count, epochs in enumerate(all_epochs):
                if 'inv_op' not in locals():
                    inv_op = make_inverse_operator(epochs.info, fwd, noise_cov)
                stc = apply_inverse_epochs(epochs, inv_op, lambda2, method)
                if 'morph' not in locals():
                    morph = compute_source_morph(stc[0], subject_from=subject, subject_to='fsaverage',
                                                 subjects_dir=pathnames['mri_root_dir'])
                for idx, item in enumerate(stc):
                    if epoch_count == 0:
                        conn_name = event_name + '_' + str(idx).zfill(3) + '_' + str(idx_time).zfill(3) + '.txt'
                    elif epoch_count == 1:
                        conn_name = (event_name + '_' + str(idx + len(all_epochs[0].events)).zfill(3) + '_' +
                                     str(idx_time).zfill(3) + '.txt')
                    if op.isfile(op.join(band_output_dir, conn_name)):
                        print(conn_name + ' exists. Moving to next...')
                        continue
                    item_fs = morph.apply(item)
                    label_ts = item_fs.extract_label_time_course(labels_parc, fs_src,
                                                                 mode='mean_flip',
                                                                 allow_empty=True)
                    con = spectral_connectivity([label_ts], method='imcoh',
                                                mode='multitaper',
                                                sfreq=epochs.info['sfreq'],
                                                fmin=fmin,
                                                fmax=fmax,
                                                faverage=True,
                                                verbose='WARNING')[0]
                    c = con[:, :, 0]
                    print(conn_name)
                    write_connectivity_matrix(c.round(decimals=6), op.join(band_output_dir, conn_name))
                    # write_connectivity_matrix(c, op.join(band_output_dir, conn_name))




"""Steps needed for processing the clean cropped files.

( 1) Create the epochs
( 1) Load the forward solution ??
( 2) Make inverse source
( 3) Make morphed source
( 4) Generate connectivity matrix
( 5) Save the connectivity matrix



import eses_spike_conn_timecourse as sct

mri_root_dir = '/Users/joshbear/clinical/meg/anat/'
subject = 'ESES_01'

events_file = '/Users/joshbear/research/ied_network/data/subjects/ESES_01/meg/data/rest_1-raw.txt'
raw_file = '/Users/joshbear/research/ied_network/data/subjects/ESES_01/meg/data/rest_1_clean-raw.fif'
noise_file = '/Users/joshbear/research/ied_network/data/subjects/ESES_01/meg/pipeline/noise-cov.fif'
fwd_file = '/Users/joshbear/research/ied_network/data/subjects/ESES_01/meg/pipeline/rest_1-fwd.fif'

t_window = 0.5  # length of window in seconds, equal to the length of desired epochs
t_start = 0.  # starting point relative to the event time in seconds

snr = 1.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'
vertices_to = [np.arange(10242), np.arange(10242)]

epochs = sct.get_epochs(raw_file, events_file, t_start=t_start, t_window=t_start+t_window, event_id=1)
noise_cov = read_cov(noise_file)
fwd = read_forward_solution(fwd_file)
inv_op = make_inverse_operator(epochs.info, fwd, noise_cov)
stc = apply_inverse_epochs(epochs, inv_op, lambda2, method)




src_file = '/Users/joshbear/research/ied_network/data/subjects/ESES_01/meg/anat/ESES_01-src.fif'
src = read_source_spaces(src_file)


stc_fs = []
morph = compute_source_morph(stc[0], subject_from=subject, subject_to='fsaverage', subjects_dir=mri_root_dir)
for item in stc:
    item_fs = morph.apply(item)
    stc_fs.append(item_fs)


labels_parc = read_labels_from_annot(
            'fsaverage', parc='HCPMMP1',
            subjects_dir=mri_root_dir)
fs_src = read_source_spaces('/Users/joshbear/research/epi_conn/' +
                            'fsaverage/anat/fsaverage-src.fif')

stc_fs = []
morph = compute_source_morph(stc[0], subject_from=subject, subject_to='fsaverage', subjects_dir=mri_root_dir)
for item in stc:
    item_fs = morph.apply(item)
    label_ts = item_fs.extract_label_time_course(labels_parc, fs_src,
                                                 mode='mean_flip',
                                                 allow_empty=True)
    stc_fs.append(label_ts)

fmin = 12.
fmax = 30.
m = []
for source in stc_fs:
    con = spectral_connectivity([source], method='imcoh',
                                mode='multitaper',
                                sfreq=epochs.info['sfreq'],
                                fmin=fmin,
                                fmax=fmax,
                                faverage=True,
                                verbose='WARNING')[0]
    c = con[:, :, 0]
    m.append(c)



freqs = [[12, 30]]
est.run_eses_spike_conn_time_series(subject='E-0223', time_start=-5, time_stop=5,
                                    freqs=freqs, time_window=0.25, time_step=0.125, event='event1', scan_type='clinical')
freqs = [[12, 30], [30, 55], [55, 80], [80, 120], [120, 250]]
est.run_eses_spike_conn_time_series(subject='ESES_11', time_start=-5, time_stop=5,
                                    freqs=freqs, time_window=0.5, time_step=0.25, event='event1', scan_type='eses')
est.run_eses_spike_conn_time_series(subject='ESES_30', time_start=-5, time_stop=5,
                                    freqs=freqs, time_window=0.25, time_step=0.125, event='event1', scan_type='eses')
freqs = [[8, 12], [12, 30], [30, 55], [55, 80], [80, 120]]
est.run_eses_spike_conn_time_series(subject='ESES_30', time_start=-5, time_stop=5,
                                    freqs=freqs, time_window=0.5, time_step=0.25, event='event1', scan_type='eses')

"""
