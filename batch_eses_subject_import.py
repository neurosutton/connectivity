import sys
sys.path.append('/Users/joshbear/research/eses_connectivity/scripts')
import preprocess
import os.path as op
import glob
import mne

# raw_dir = '/Volumes/bear_research/eses_connectivity/data/subjects'
# raw_dir = None
# study_dir = '/Users/joshbear/research/ied_network/'

# subjects = ['ESES_01', 'ESES_03', 'ESES_05', 'ESES_10', 'ESES_11', 'ESES_30', 'ESES_34', 'ESES_35']

# 'ESES_01', 'ESES_05', 'ESES_10', 'ESES_11', 'ESES 30'

def perform_group_preprocessing(subjects, raw_dir, study_dir, bad_channels_file=None):
    for subject in subjects:
        subj = preprocess.Subject(subject, raw_dir, study_dir)
        subj.make_noise_covariance_file()
        # subj.preprocess_anat()
        # subj.import_meg_runs()
        # subj.filter_raw(l_freq=1, h_freq=None, maxwell_filter=False, bad_channels_file=bad_channels_file)
        # subj.remove_artifacts()
        # subj.make_forward_solution()

def get_subject_runs(subject, raw_dir=None):
    """
    Given a subject ID, return a list of the MEG recordings.

    """
    meg_dir = op.join(raw_dir, subject, 'ESES')
    all_runs = {}
    all_runs['files'] = []
    all_runs['sizes'] = []

    for file in glob.iglob(meg_dir + '/**/c,rfhp0.1Hz', recursive=True):
        all_runs['files'].append(file)
        all_runs['sizes'].append(op.getsize(file))

    return all_runs


def list_subject_runs(all_runs):
    for idx in range(len(all_runs['files'])):
        print(str(idx) + ': ', end='')
        path_names = all_runs['files'][idx].split('/')[-5:]
        print(op.join(*path_names), end=' ')
        print('(filesize: ' + str(all_runs['sizes'][idx]) + ')')


def get_run_assignment_dictionary(all_runs=None, empty_room=None, rest_1=None, rest_2=None):
    """ Assign file paths from all_runs into a dictionary to track files for import."""
    run_dict = {'empty_room' : None, 'rest_1' : None, 'rest_2' : None}
    if type(empty_room) is int:
        run_dict['empty_room'] = all_runs['files'][empty_room]
    elif type(empty_room) is str:
        run_dict['empty_room'] = empty_room
    if type(rest_1) is int:
        run_dict['rest_1'] = all_runs['files'][rest_1]
    elif type(rest_1) is str:
        run_dict['rest_1'] = rest_1
    if type(rest_2) is int:
        run_dict['rest_2'] = all_runs['files'][rest_2]
    elif type(rest_2) is str:
        run_dict['rest_2'] = rest_2
    return run_dict

def import_meg_runs(run_dict, study_dir, subject):
    """Import the MEG runs defined in a dictionary containing paths to each file type."""
    fname_empty_room = op.join(study_dir, 'data', subject, 'meg', 'data', 'empty_room-raw.fif')
    # fname_empty_room = op.join(self.res_data, 'empty_room-raw.fif')
    fname_rest_1 = op.join(study_dir, 'data', subject, 'meg', 'data', 'rest_1-raw.fif')
    fname_rest_2 = op.join(study_dir, 'data', subject, 'meg', 'data', 'rest_2-raw.fif')
    dir_data = op.join(study_dir, 'data', subject, 'meg', 'data')

    for run_type, run_file in run_dict.items():
        run_path = op.join('/', *run_file.split('/')[0:-1])
        if run_type == 'empty_room':
            if op.isfile(fname_empty_room):
                print('Empty room recording has already been imported.')
                continue
            else:
                print("attempting to load empty room")
                raw = mne.io.read_raw_bti(pdf_fname=run_file,
                                          config_fname=run_path + '/config',
                                          head_shape_fname=None)
                print("loaded empty room")
                raw.save(op.join(dir_data, run_type + '-raw.fif'))
                print("saved empty room")
                continue
        if run_type == 'rest_1':
            if op.isfile(fname_rest_1):
                print('rest_1 recording has already been imported.')
                continue
        if run_type == 'rest_2':
            if op.isfile(fname_rest_2):
                print('rest_2 recording has already been imported.')
                continue
        if run_type is not 'empty_room':
            raw = mne.io.read_raw_bti(pdf_fname=run_file,
                                      config_fname=run_path + '/config',
                                      head_shape_fname=run_path + '/hs_file')
            raw.save(op.join(dir_data, run_type + '-raw.fif'))


"""
import os.path as op
import sys, glob, importlib
import batch_eses_subject_import as esi
import mne
import glob

raw_dir = '/Volumes/bear_research/eses_connectivity/data/subjects'
study_dir = '/Users/joshbear/research/ied_network/'

subject = 'ESES_39'
all_runs = esi.get_subject_runs(subject, raw_dir)
esi.list_subject_runs(all_runs)

run_dict = esi.get_run_assignment_dictionary(all_runs=all_runs, empty_room=1, rest_1=3, rest_2=2)
esi.import_meg_runs(run_dict, study_dir, subject)


importlib.reload(bat)

subjects = ['ESES_01', 'ESES_03', 'ESES_05', 'ESES_10', 'ESES_11', 'ESES_30', 'ESES_34', 'ESES_35']
bat.perform_group_preprocessing(subjects, raw_dir, study_dir, bad_channels_file='/Users/joshbear/research/ied_network/data/eses_subjects_bad_channels.csv')
"""
