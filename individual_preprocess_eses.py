"""Simple script to preprocess data from the clinical subject scans.

This essentially modifies the ESES connetivity script preprocess.py
to allow preprcoessing of clinical scans. The main difference for these
individuals is that there are more resting scans, no empty room scan, and
no attempt at a cross hair as well as having a different directory structure.

Given these limitations, a noise covariance is taken from ESES_01 rather than
being made separately for the runs.

The steps that this process works through are:
( 1) Load subject: subj = preprocess.Subject('Subj_ID')
( 2) Load and pre-process the anatomy files: subj.preprocess_anat()
( 3) Import the MEG data: subj.import_meg_runs()
( 4) Perform co-registration from terminal: mne coreg
     Save the file as rest-trans.fif file in the 'data' folder.
( 5) Perform initial cleaning: subj.filter_raw()
( 6) Perform artifact removal / final filtering: subj.remove_artifacts()

Afterwards, a separate setup will need to be performed to create the analyses.
"""

import os
import os.path as op
import glob
import mne
import numpy as np
from mne.preprocessing import ICA

# Define which subject to look for. Could update to be interactive later.
subject = 'E-0215'

meg_dir = op.join('/Users/joshbear/research/eses_connectivity/data/subjects/', subject,
                  'meg', 'data')
study_dir = '/Users/joshbear/research/ied_network/'
subject_dir = op.join(study_dir, 'data', 'subjects')
mri_root_directory = '/Users/joshbear/clinical/meg/anat/'
mri_dir = op.join(mri_root_directory, subject)
results_dir = op.join(subject_dir, subject, 'meg')
noise_cov_file = ('/Users/joshbear/research/eses_connectivity/data/subjects/' +
                  'ESES_01/meg/pipeline/noise-cov.fif')  # none for clinical

if op.isdir(op.join(subject_dir, subject)) is False:
    print("subject_dir (" + subject_dir +
          ") does not exist and will be created.")
    os.mkdir(op.join(subject_dir, subject))
if op.isdir(results_dir) is False:
    print("results_folder (" + results_dir +
          ") does not exist and will be created.")
    os.mkdir(results_dir)
res_anat = op.join(results_dir, 'anat')
if op.isdir(res_anat) is False:
    print('Creating results anatomy folder.')
    os.mkdir(res_anat)
res_data = op.join(results_dir, 'data')
if op.isdir(res_data) is False:
    print('Creating results data folder.')
    os.mkdir(res_data)
res_pipe = op.join(results_dir, 'pipeline')
if op.isdir(res_pipe) is False:
    print('Creating results pipeline data folder.')
    os.mkdir(res_pipe)
res_out = op.join(results_dir, 'output')
if op.isdir(res_out) is False:
    print('Creating results output data folder.')
    os.mkdir(res_out)


def get_subject_runs(subject):
    """
    Given a subject ID, return a list of the MEG recordings.

    """
    all_runs = []
    for file in glob.iglob(meg_dir + '/**/c,rfhp0.1Hz', recursive=True):
        all_runs.append(file)
    all_runs = sorted(all_runs)
    # There should be 4 (empty, two runs with movie, and a fixation cross)
    # However, the fixation cross has been a bust, so not including here.
    return all_runs


def get_imported_runs(subject):
    """Find the raw imported runs for given subject."""
    all_runs = []
    for file in glob.iglob(res_data + '/rest_??_raw.fif', recursive=False):
        all_runs.append(file)
    return all_runs


def import_meg_runs(subject):
    """Import the MEG runs here."""
    raw_runs = get_subject_runs(subject)
    for idx, run in enumerate(raw_runs):
        save_as = 'rest_' + str(idx).zfill(2) + '_raw.fif'
        if op.isfile(op.join(res_data, save_as)):
            print('File already imported: ' + save_as)
            print('Moving to next file...')
            continue
        run_name = run.split('/')[-1]
        run_path = run[0:-len(run_name)]
        raw = mne.io.read_raw_bti(pdf_fname=run,
                                  config_fname=run_path + 'config',
                                  head_shape_fname=run_path + 'hs_file')
        raw.save(op.join(res_data, save_as))


def preprocess_anat(subject):
    """
    Generate source space and BEM model/solution.

    Will check for source file (-src.fif), BEM file (-bem.fif) and
    BEM solution (-bem-sol.fif). If these files do not exist, they
    will be created and saved in the /anat folder.

    Args
    ----
        None

    Returns
    -------
        None

    """
    src_file = op.join(res_anat, '%s-src.fif' % subject)
    vol_src_file = op.join(res_anat, '%s-vol-src.fif' % subject)
    bem_file = op.join(res_anat, '%s-bem.fif' % subject)
    bem_sol_file = op.join(res_anat, '%s-bem-sol.fif' % subject)

    if op.isfile(bem_file):
        print('Found an existing BEM file (%s).' % bem_file)
    else:
        print('Creating BEM using make_bem_model with conductivity '
              '= 0.3.\nFile will be saved as: %s' % bem_file)
        if op.isdir(op.join(mri_root_directory, subject, 'bem')) is False:
            print('Watershed BEM for subject not yet made. Making it so...')
            mne.bem.make_watershed_bem(subject,
                                       subjects_dir=mri_root_directory)
        # puts stuff in this directory:
        # /Users/joshbear/clinical/meg/anat/E-0191/bem/watershed
        model = mne.make_bem_model(subject, conductivity=[0.3],
                                   subjects_dir=mri_root_directory)
        mne.write_bem_surfaces(bem_file, model)

    if op.isfile(src_file):
        print('Found an existing source file (%s).' % src_file)
    else:
        print('Creating source space using setup_source_space).\n'
              'File will be saved as: %s' % src_file)
        src = mne.setup_source_space(subject, spacing='ico5',
                                     subjects_dir=mri_root_directory)

        # Alternatively, if I end up using a volume source space...
        fname_aseg = op.join(mri_root_directory, subject, 'mri', 'aseg.mgz')
        vol_src = mne.setup_volume_source_space(subject, mri=fname_aseg, pos=10.0, bem=bem_file,
                                                subjects_dir=mri_root_directory)

        mne.write_source_spaces(src_file, src)
        mne.write_source_spaces(vol_src_file, vol_src)

    if op.isfile(bem_sol_file):
        print('Found an existing BEM solution file (%s).' % bem_sol_file)
    else:
        if 'model' not in vars():
            print('Attempting to load prior *bem.fif...')
            model = mne.read_bem_surfaces(bem_file)

        print('Creating BEM solution using make_bem_model with '
              'connectivity = 0.3.\nFile will be saved as: %s'
              % bem_sol_file)
        model_solution = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_sol_file, model_solution)
    print('All steps in preprocess_anat() complete. '
          'Next, you might want to call preprocess_raw()')


def filter_raw_runs(subject):
    """Initial filtering of the raw runs."""
    files = get_imported_runs(subject)
    noise_cov = mne.read_cov(noise_cov_file)

    for file in files:
        clean_name = 'clean_' + file.split('/')[-1]
        if op.isfile(op.join(res_data, clean_name)):
            print(clean_name + ' already exists. Next...')
            continue
        else:
            raw = mne.io.read_raw_fif(file, preload=True)
            raw.pick_types(meg=True, ecg=True, exclude='bads')
            clean = mne.preprocessing.maxwell_filter(raw, st_only=True,
                                                     st_duration=10)
            clean.filter(l_freq=1, h_freq=None)
            """
            clean.info['bads'] = ['MEG 229', 'MEG 156']  # not in noise cov
            ica = ICA(n_components=0.95, method='fastica', random_state=0,
                      noise_cov=noise_cov, max_iter=500)
            clean.pick_types(meg=True, ecg=True, exclude='bads')
            ica.fit(clean, decim=3,
                    reject=dict(mag=5e-12, grad=4000e-13),
                    verbose='warning')
            ecg_inds, ecg_scores = ica.find_bads_ecg(clean,
                                                     ch_name='EEG 001',
                                                     method='ctps')
            ica.exclude += ecg_inds
            print('Applying ICA.')
            ica.apply(clean)
            print('Filtering the bandpass and downsampling.')
            clean = clean.filter(l_freq=1, h_freq=55)
            clean.resample(120)
            print('Saving the cleaned data.')
            """
            clean.save(op.join(res_data, clean_name))


import_meg_runs(subject)
preprocess_anat(subject)
filter_raw_runs(subject)
