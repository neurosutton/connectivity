"""Run spike_conn processing on all events."""

import spike_conn as sc
import os.path as op
import sys
from os import mkdir
from mne.minimum_norm import apply_inverse_raw

mri_root_dir = '/Users/joshbear/clinical/meg/anat/'
subjects_root_dir = '/Users/joshbear/research/eses_connectivity/data/subjects'

# subject = input('Subject name: ')
subject = input('Subject ID: ')

raw_dir = op.join(subjects_root_dir, subject, 'meg', 'data')
anat_dir = op.join(subjects_root_dir, subject, 'meg', 'anat')
pipeline_dir = op.join(subjects_root_dir, subject, 'meg', 'pipeline')
output_dir = op.join(subjects_root_dir, subject, 'meg', 'output')

if op.isdir(raw_dir) is False:
    print('Subject <' + subject + '>: Could not find data folder. Exiting.')
    sys.exit()

run = input('MEG Recording Run: ')
events_file = run + '_clean-raw_events.txt'
event_file_path = op.join(pipeline_dir, events_file)
events = sc.make_events_dict(op.join(pipeline_dir, events_file))

print('Subject and run both appear to be valid selections. Proceeding.')

fname_clean = op.join(raw_dir, run + '_clean-raw.fif')
fname_empty_room = op.join(raw_dir, 'empty_room-raw.fif')
fname_trans = op.join(raw_dir, run + '-trans.fif')
fname_bem_sol = op.join(anat_dir, subject + '-bem-sol.fif')
fname_src = op.join(anat_dir, subject + '-src.fif')
fname_noise_cov = op.join(pipeline_dir, 'noise-cov.fif')

snr = 1.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

# con = create_connectivity_matrix()

"""
Some things I need to pass to create_connectivity_matrix:

For get_raw_segment:
    start (time)
    stop (time)

For get_forward_solution:
    fname_src
    fname_trans
    fname_bem_sol

For get_inverse_operator:
    fname_noise_cov

get_morphed_source:
    mri_root_dir

get_connectivity_matrix:
    sfreq
    fmin
    fmax
    mode
    method
    parc

write_connectivity_matrix:
    fname_conn
"""

print('The following events were identified in given file:')
sc.print_events(events)

event_type = input('Which event, or \'all\', to process? : ')
print('Will process event_type: ' + event_type)
conn_output_dir = op.join(output_dir, event_type + '_spike_conn')

if op.isdir(conn_output_dir) is False:
    print('Making folder for ' + event_type + ' conn output.')
    mkdir(conn_output_dir)

# should add option to define the start and stop times ...
for idx, event in enumerate(events):
    if event['type'] == event_type:
        print('Processing: ' + str(event))
        conn_name = event_type + '_bp_12_30_' + str(idx).zfill(3) + 'a.txt'
        start = event['start'] - 1.05  # - 1.05
        stop = event['start'] - 0.05  # - 0.05
        raw = sc.get_raw_segment(fname_clean, start=start, stop=stop)
        fwd = sc.get_forward_solution(raw, fname_src, fname_trans,
                                      fname_bem_sol)
        inv_op = sc.get_inverse_operator(raw, fwd, fname_noise_cov)
        stc = apply_inverse_raw(raw, inv_op, lambda2, method)
        stc_to = sc.get_morphed_source(stc, mri_root_dir, subject)
        con = sc.get_connectivity_matrix(
            stc_to,
            raw.info['sfreq'],
            fmin=12,
            fmax=30,
            mode='mean_flip',
            method='imcoh',
            parc='HCPMMP1')
        sc.write_connectivity_matrix(con, op.join(conn_output_dir, conn_name))
        # and do it again for part B, should make this prettier sometime...

        conn_name = event_type + '_bp_12_30_' + str(idx).zfill(3) + 'b.txt'
        start = event['start']  # formerly: no extra time
        stop = event['start'] + 1  # formerly: + 1
        raw = sc.get_raw_segment(fname_clean, start=start, stop=stop)
        fwd = sc.get_forward_solution(raw, fname_src, fname_trans,
                                      fname_bem_sol)
        inv_op = sc.get_inverse_operator(raw, fwd, fname_noise_cov)
        stc = apply_inverse_raw(raw, inv_op, lambda2, method)
        stc_to = sc.get_morphed_source(stc, mri_root_dir, subject)
        con = sc.get_connectivity_matrix(
            stc_to,
            raw.info['sfreq'],
            fmin=12,
            fmax=30,
            mode='mean_flip',
            method='imcoh',
            parc='HCPMMP1',
            subjects_dir=mri_root_dir)
        sc.write_connectivity_matrix(con, op.join(conn_output_dir, conn_name))
        """
        # third time...
        conn_name = event_type + '_bp_12_30_' + str(idx).zfill(3) + 'c.txt'
        start = event['stop']  # formerly: no extra time
        stop = event['stop'] + 1  # formerly: + 1
        raw = sc.get_raw_segment(fname_clean, start=start, stop=stop)
        fwd = sc.get_forward_solution(raw, fname_src, fname_trans,
                                      fname_bem_sol)
        inv_op = sc.get_inverse_operator(raw, fwd, fname_noise_cov)
        stc = apply_inverse_raw(raw, inv_op, lambda2, method)
        stc_to = sc.get_morphed_source(stc, mri_root_dir, subject)
        con = sc.get_connectivity_matrix(
            stc_to,
            raw.info['sfreq'],
            fmin=12,
            fmax=30,
            mode='mean_flip',
            method='imcoh',
            parc='HCPMMP1',
            subjects_dir=mri_root_dir)
        sc.write_connectivity_matrix(con, op.join(conn_output_dir, conn_name))
        """
print('\nAll processing completed!\n')
