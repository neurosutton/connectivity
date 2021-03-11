"""Module to analyze MEG resting state data using MNE.

This module provides a series of functions and a new data object RestSubject
designed to streamline the processing of global connectivity in resting state
MEG recordings.

Suggested Workflow
------------------
( 1) Load subject: subj = preprocess.Subject('Subj_ID')
( 2) Load and pre-process the anatomy files: subj.preprocess_anat()
     - This tends to fail at the watershed.
     - From terminal: mne watershed_bem -s SUBJ_ID
( 3) Import the MEG data: subj.import_meg_runs()
( 4) Perform co-registration from terminal: mne coreg
     Save the file as rest-trans.fif file in the 'data' folder.
( 5) Perform initial cleaning: subj.filter_raw()
( 6) Perform artifact removal / final filtering: subj.remove_artifacts()
( 7) Make forward solution: make_forward_solution()
( 8) Make inverse source: make_inverse_source()
( 9) Make morphed source: make_morphed_source(fname=STC file)

"""

import os
import os.path as op
import glob
import mne
import numpy as np
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import matplotlib as mpl
from surfer import Brain
import pickle

mri_root_directory = '/Users/joshbear/clinical/meg/anat/'


def get_subject_folders(subject, raw_dir=None, study_dir=None):
    """
    Given a subject ID, return paths to all pertinent folders.

    Args
    ----
        subject:    Subject ID as string, used to generate paths

    Returns
    -------
        Dictionary containing the MEG directory 'meg', the anatomy
        directory 'anat', the results directory 'results', and a
        list of all runs 'runs'

    """
    if study_dir is None:
        study_dir = '/Users/joshbear/research/eses_connectivity/'
    subject_dir = op.join(study_dir, 'data', 'subjects')
    # Check if we are using a clinical or research scan
    if subject[0:2] == 'E-':  # clinical scan
        meg_dir = op.join('/Users/joshbear/clinical/meg/data', subject,
                          'EPILEPSY24')
    else:  # assume a research scan
        if raw_dir is None:
            meg_dir = op.join(subject_dir, subject, 'ESES')
        else:
            meg_dir = op.join(raw_dir, subject, 'ESES')
    mri_dir = op.join(mri_root_directory, subject)
    results_dir = op.join(subject_dir, subject, 'meg')

    if op.isdir(op.join(subject_dir, subject)) is False:
        print("subject folder (" + op.join(subject_dir, subject) +
              ") does not exist and will be created.")
        os.mkdir(op.join(subject_dir, subject))
    if op.isdir(meg_dir) is False:
        print("MEG data directory (" + meg_dir + ") does not exist.")
        print("Exiting on getUserFolders()")
        return  # make sure calling functions can cope with return = None   UNCOMMENT THIS IN THE FUTURE!!!!!
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

    return {'meg': meg_dir,
            'anat': mri_dir,
            'results': results_dir,
            'res_anat': res_anat,
            'res_data': res_data,
            'res_pipe': res_pipe,
            'res_out': res_out}


def get_subject_runs(subject, raw_dir=None, study_dir=None):
    """
    Given a subject ID, return a list of the MEG recordings.

    """
    meg_dir = get_subject_folders(subject, raw_dir, study_dir)['meg']
    all_runs = []

    for file in glob.iglob(meg_dir + '/**/c,rfhp0.1Hz', recursive=True):
        all_runs.append(file)

    all_runs = sorted(all_runs)
    # There should be 4 (empty, two runs with movie, and a fixation cross)
    # However, the fixation cross has been a bust, so not including here.
    return {'empty_room': all_runs[0],
            'rest_1': all_runs[1],
            'rest_2': all_runs[2]}


def convert_chname_to_mne(ch):
    """Small helper function for fixing sensor names."""
    if ch[0] != 'A':
        return ch
    ch_new = 'MEG ' + ch[1:].zfill(3)
    return ch_new


def fix_channel_names(file):
    """Change from A# to MEG 00# format."""
    raw = mne.io.read_raw_fif(file, preload=True)
    for ch in raw.info['ch_names']:
        ch_new = convert_chname_to_mne(ch)
        raw.info['ch_names'][raw.info['ch_names'].index(ch)] = ch_new
    for idx, ch in enumerate(raw.info['chs']):
        ch_new = convert_chname_to_mne(ch['ch_name'])
        raw.info['chs'][idx]['ch_name'] = ch_new
    return raw


def load_conoutput(full_path):
    """Load a saved ConOutput file."""
    if op.isfile(full_path) is False:
        print('Unable to found that object. This requires a full path.')
    else:
        with open(full_path, 'rb') as file:
            return pickle.load(file)


def get_bad_channels_from_csv(file):
    """Reads bad channels from a CSV where the first column contains subject IDs."""
    bads = {}
    try:
        bads_in = open(file, 'r', encoding='utf-8-sig')
    except FileNotFoundError:
        print('Tried to open ' + file_path + '\nNo such file. Exiting.')
        sys.exit()
    lines = bads_in.readlines()
    for row in lines:
        bads_list = row.strip().split(',')
        if len(bads_list) == 0:
            continue
        subject = bads_list[0]
        bads_list = [chan for chan in bads_list[1:] if chan != '']
        bads[subject] = bads_list
    bads_in.close()
    return bads


class Subject(object):
    """
    A collection of variables for a single-subject resting state scan.

    Args
    ----
        subject (str): Subject ID (used for finding folders)

    Attributes
    ----------
        data (str): Directory for MEG data.
        anat (str): Directory for anatomy data.
        res (str) : Directory for output from class functions.
        runs (list) : A list of strings, sorted alphabetically, containing
            full paths to all data files identified on a glob search.

    """

    def __init__(self, subject, raw_dir=None, study_dir=None):
        """Initialize function."""
        self.subject = subject

        # Data Root Directories
        self.mri_dir = get_subject_folders(subject, raw_dir, study_dir)['anat']
        self.meg_dir = get_subject_folders(subject, raw_dir, study_dir)['meg']
        self.res_dir = get_subject_folders(subject, raw_dir, study_dir)['results']

        # Results Directories (for processing and analysis output)
        self.res_anat = get_subject_folders(subject, raw_dir, study_dir)['res_anat']
        self.res_data = get_subject_folders(subject, raw_dir, study_dir)['res_data']
        self.res_pipe = get_subject_folders(subject, raw_dir, study_dir)['res_pipe']
        self.res_out = get_subject_folders(subject, raw_dir, study_dir)['res_out']

        # Filename variables for finding everything.
        self.fname_empty_room = op.join(self.res_data, 'empty_room-raw.fif')
        self.fname_rest_1 = op.join(self.res_data, 'rest_1-raw.fif')
        self.fname_rest_2 = op.join(self.res_data, 'rest_2-raw.fif')
        self.fname_clean_empty_room = op.join(self.res_data,
                                              'empty_room_clean-raw.fif')
        self.fname_clean_rest_1 = op.join(self.res_data,
                                          'rest_1_clean-raw.fif')
        self.fname_clean_rest_2 = op.join(self.res_data,
                                          'rest_2_clean-raw.fif')
        self.fname_noise_cov = op.join(self.res_pipe, 'noise-cov.fif')
        self.fname_trans = op.join(self.res_data, self.subject + '-trans.fif')
        self.fname_fwd_rest_1 = op.join(self.res_pipe, 'rest_1-fwd.fif')
        self.fname_fwd_rest_2 = op.join(self.res_pipe, 'rest_2-fwd.fif')
        self.fname_bem = op.join(self.res_anat, self.subject + '-bem.fif')
        self.fname_bem_sol = op.join(self.res_anat,
                                     self.subject + '-bem-sol.fif')
        self.fname_src = op.join(self.res_anat, '%s-src.fif' % self.subject)

        # Get the MEG recordings
        self.meg_runs = get_subject_runs(subject, raw_dir, study_dir)

        # Important variables that will need to be saved individually.
        self.bads = []  # for defining the bad channels for all runs

    def import_meg_runs(self):
        """Import the MEG runs here."""
        raw_runs = self.meg_runs  # raw_runs = get_subject_runs(self.subject)
        fname_empty_room = op.join(self.res_data, 'empty_room-raw.fif')
        fname_rest_1 = op.join(self.res_data, 'rest_1-raw.fif')
        fname_rest_2 = op.join(self.res_data, 'rest_2-raw.fif')

        for run_type, run_file in raw_runs.items():
            run_name = run_file.split('/')[-1]
            run_path = run_file[0:-len(run_name)]

            # a little logic to ensure we don't import files multiple times
            if run_type == 'empty_room':
                if op.isfile(fname_empty_room):
                    print('Empty room recording has already been imported.')
                    continue
                else:
                    print("attempting to load empty room")
                    raw = mne.io.read_raw_bti(pdf_fname=run_file,
                                              config_fname=run_path + 'config',
                                              head_shape_fname=None)
                    print("loaded empty room")
                    raw.save(op.join(self.res_data, run_type + '-raw.fif'))
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

            raw = mne.io.read_raw_bti(pdf_fname=run_file,
                                      config_fname=run_path + 'config',
                                      head_shape_fname=run_path + 'hs_file')
            raw.save(op.join(self.res_data, run_type + '-raw.fif'))

    def filter_raw(self, l_freq=None, h_freq=None, maxwell_filter=False, bad_channels_file=None):
        """
        Perform initial filtering of the data.

        This is limited to loading the files, marking the bad channels,
        selecting the appropriate sensors, performing the maxwell filter,
        and then applying a high-pass filter of 1 Hz to prepare for the
        subsequent step that will include ICA artifact removal.

        Output
        ------
        - saves a cleaned file for both rest runs and the empty room
        - does not return anything

        """
        files = [[self.fname_rest_1, self.fname_clean_rest_1],
                 [self.fname_rest_2, self.fname_clean_rest_2],
                 [self.fname_empty_room, self.fname_clean_empty_room]]

        for file, clean_file in files:
            if op.isfile(clean_file) is False and op.isfile(file) is True:
                if 'empty_room' in file:
                    raw = fix_channel_names(file)
                else:
                    raw = mne.io.read_raw_fif(file, preload=True)
                # raw.info['bads'] = self.bads
                """ The following logic removes the bad channels if they are
                not present in the ch_names list. This was necessary due to the
                empty room recording work-around. Because the data are brought
                into FieldTrip and exported to FIF, the usual channel renaming
                does not occur, making an inconsistency. Ideally this will work
                seamlessly once the problem is corrected without changing.
                """
                for bad in self.bads:
                    if bad in raw.info['ch_names'] is True:
                        raw.info['bads'].append(bad)
                # Separately, will add the bads from a file if provided.
                if bad_channels_file is not None:
                    bads = get_bad_channels_from_csv(bad_channels_file)
                    for bad in bads:
                        if bad in raw.info['ch_names'] is True:
                            if bad in raw.info['bads'] is False:
                                raw.info['bads'].append(bad)
                raw.pick_types(meg=True, ecg=True, exclude='bads')
                # no maxwell without a head shape file
                if 'empty_room' in file:
                    clean = raw.filter(l_freq=l_freq, h_freq=h_freq)
                else:
                    if maxwell_filter is True:
                        clean = mne.preprocessing.maxwell_filter(raw, st_only=True,
                                                                 st_duration=10)
                        clean.filter(l_freq=l_freq, h_freq=h_freq)
                    else:
                        clean = raw.filter(l_freq=l_freq, h_freq=h_freq)
                clean.notch_filter(60)
                clean.save(clean_file)
            else:
                print(file + ' cleaned or could not be located.')

    def make_noise_covariance_file(self):
        if op.isfile(self.fname_noise_cov) is False:
            print('No noise covariance. Generating from empty room data.')
            empty_room = mne.io.read_raw_fif(self.fname_clean_empty_room)
            noise_cov = mne.compute_raw_covariance(empty_room,
                                                   tmin=0,
                                                   tmax=None)
            mne.write_cov(self.fname_noise_cov, noise_cov)

    def remove_artifacts(self, fmin=1, fmax=55):
        """
        Noise covariance and ICA removal.

        This will implement the automated methods of identifying artifactual
        components from EOG and ECG correlations. In addition, it will filter
        the final file down to the frequency band desired for further analysis.
        This is the final step in preprocessing the MEG data.

        Arguments
        ---------
        fmin     : Setting for high-pass filter.
        fmax     : Setting for low-pass filter.

        """
        rest_files = [self.fname_clean_rest_1,
                      self.fname_clean_rest_2]
        empty_room_file = self.fname_clean_empty_room
        noise_cov_file = self.fname_noise_cov

        if op.isfile(noise_cov_file) is False:
            print('No noise covariance. Generating from empty room data.')
            empty_room = mne.io.read_raw_fif(empty_room_file)
            noise_cov = mne.compute_raw_covariance(empty_room,
                                                   tmin=0,
                                                   tmax=None)
            mne.write_cov(noise_cov_file, noise_cov)
        else:
            noise_cov = mne.read_cov(noise_cov_file)

        for file in rest_files:
            if op.isfile(file) is True:
                ica = ICA(n_components=0.95, method='fastica', random_state=0,
                          noise_cov=noise_cov, max_iter=500)
                clean = mne.io.read_raw_fif(file, preload=True)
                clean.pick_types(meg=True, ecg=True, exclude=self.bads)
                ica.fit(clean, decim=3,
                        reject=dict(mag=4e-12, grad=4000e-13),
                        verbose='warning')

                # for ECG artifacts
                print('Calculating ECG-correlated components.')
                ecg_inds, ecg_scores = ica.find_bads_ecg(clean,
                                                         ch_name='EEG 001',
                                                         method='ctps')
                # for EOG artifacts
                # print('Calculating EOG-correlated components.')
                # eog_inds, eog_scores = ica.find_bads_eog(clean,
                #                               ch_name='VEOG', threshold=3.0)
                ica.exclude += ecg_inds
                # ica.exclude += eog_inds

                print('Applying ICA.')
                ica.apply(clean)
                print('Filtering the bandpass and downsampling.')
                clean = clean.filter(l_freq=fmin, h_freq=fmax)
                # clean.resample(120)
                print('Saving the cleaned data.')
                clean.save(file, overwrite=True)

        clean = mne.io.read_raw_fif(empty_room_file, preload=True)
        clean = clean.filter(l_freq=fmin, h_freq=fmax)
        clean.resample(120)
        clean.save(empty_room_file, overwrite=True)

    def preprocess_anat(self):
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
        src_file = op.join(self.res_anat, '%s-src.fif' % self.subject)
        bem_file = op.join(self.res_anat, '%s-bem.fif' % self.subject)
        bem_sol_file = op.join(self.res_anat, '%s-bem-sol.fif' % self.subject)

        if op.isfile(src_file):
            print('Found an existing source file (%s).' % src_file)
        else:
            print('Creating source space using setup_source_space).\n'
                  'File will be saved as: %s' % src_file)
            src = mne.setup_source_space(self.subject, spacing='ico5',
                                         subjects_dir=mri_root_directory)
            mne.write_source_spaces(src_file, src)
        if op.isfile(bem_file):
            print('Found an existing BEM file (%s).' % bem_file)
        else:
            print('Creating BEM using make_bem_model with conductivity '
                  '= 0.3.\nFile will be saved as: %s' % bem_file)
            if not op.isdir(op.join(mri_root_directory, self.subject, 'bem', 'watershed', 'ws')):
                mne.bem.make_watershed_bem(self.subject,
                                           subjects_dir=mri_root_directory)
            # puts stuff in this directory:
            # /Users/joshbear/clinical/meg/anat/E-0191/bem/watershed
            model = mne.make_bem_model(self.subject, conductivity=[0.3],
                                       subjects_dir=mri_root_directory)
            mne.write_bem_surfaces(bem_file, model)
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

    def crop_run(self, fname, tmin=0, tmax=300):
        """Crop a file."""
        try:
            raw = mne.io.read_raw_fif(op.join(self.res_data, fname))
        except FileNotFoundError:
            print('Could not load file: ' + fname)
            return
        raw.crop(tmin, tmax)
        raw.save(op.join(self.res_data,
                 fname[:-8] + '-t' + str(tmin) + '-' + str(tmax) + '-raw.fif'))

    def make_forward_solution(self, fname=''):
        """Calculate forward solution for identified file or both rest runs."""
        if fname == '':
            files = [[self.fname_clean_rest_1, self.fname_fwd_rest_1],
                     [self.fname_clean_rest_2, self.fname_fwd_rest_2]]
        else:
            files = [[op.join(self.res_data, fname),
                      op.join(self.res_pipe, fname[:-8] + '-fwd.fif')]]

        src = mne.read_source_spaces(self.fname_src)

        for file, fwd_file in files:
            if op.isfile(fwd_file):
                print(f'Forward solution \'{fwd_file}\' already exists.')
                continue
            if op.isfile(file) is False:
                print(file + ' could not be found. Moving on.')
                continue
            fwd = mne.make_forward_solution(file,
                                            self.fname_trans,
                                            src,
                                            self.fname_bem_sol,
                                            mindist=5.0,
                                            meg=True,
                                            eeg=False,
                                            n_jobs=1)
            mne.write_forward_solution(fwd_file, fwd)

    def make_inverse_source(self, fname='', method='dSPM'):
        """Calculate inverse source estimate."""
        if fname == '':
            files = [[self.fname_clean_rest_1, self.fname_fwd_rest_1],
                     [self.fname_clean_rest_2, self.fname_fwd_rest_2]]
        else:
            files = [[op.join(self.res_data, fname),
                      op.join(self.res_pipe, fname[:-8] + '-fwd.fif')]]
        snr = 1.0
        lambda2 = 1.0 / snr ** 2
        cov = mne.read_cov(self.fname_noise_cov)

        for file, fwd_file in files:
            if op.isfile(file) is False:
                print(file + ' could not be found. Moving on.')
                continue
            raw = mne.io.read_raw_fif(file)
            fwd = mne.read_forward_solution(fwd_file)
            inv_op = mne.minimum_norm.make_inverse_operator(raw.info,
                                                            fwd,
                                                            cov,
                                                            depth=None,
                                                            fixed=False)
            stc = mne.minimum_norm.apply_inverse_raw(raw,
                                                     inv_op,
                                                     lambda2,
                                                     method)
            stc.save(op.join(self.res_pipe, file.split('/')[-1][:-8]))
        return stc

    def make_morphed_source(self, fname):
        """Send inverse estimate and get a version morphed to FSAvg."""
        vertices_to = [np.arange(10242), np.arange(10242)]
        file = op.join(self.res_pipe, fname.split('/')[-1].split('.')[0])
        stc = mne.read_source_estimate(file)
        stc_to = mne.morph_data(self.subject, 'fsaverage', stc, n_jobs=1,
                                grade=vertices_to,
                                subjects_dir=mri_root_directory)
        stc_to.save(file + '-morphed')
        return stc_to

    def get_connectivity_matrix(self, stc_file, sfreq, fmin=4, fmax=24,
                                mode='mean_flip', method='imcoh',
                                parc='HCPMMP1'):
        """Provide an inverse source estimate and return connectivity."""
        fs_src = mne.read_source_spaces('/Users/joshbear/research/epi_conn/' +
                                        'fsaverage/anat/fsaverage-src.fif')
        labels_parc = mne.read_labels_from_annot(
                'fsaverage', parc=parc,
                subjects_dir=mri_root_directory
            )
        stc = mne.read_source_estimate(op.join(self.res_dir, stc_file))
        label_ts = stc.extract_label_time_course(labels_parc, fs_src,
                                                 mode=mode,
                                                 allow_empty=True)
        label_ts = [label_ts]
        con = ConOutput(*mne.connectivity.spectral_connectivity(
                        label_ts, method=method, mode='multitaper',
                        sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True,
                        mt_adaptive=True, n_jobs=1))
        return con


class ConOutput(object):
    """
    A collection of variables for a single-subject resting state scan.

    Args
    ----
        subject (str): Subject ID (used for finding folders)

    Attributes
    ----------
        data (str): Directory for MEG data.
        anat (str): Directory for anatomy data.
        res (str) : Directory for output from class functions.
        runs (list) : A list of strings, sorted alphabetically, containing
            full paths to all data files identified on a glob search.

    """

    def __init__(self, con, freqs, times, n_epochs, n_tapers):
        """Initialize function."""
        self.con = con
        self.freqs = freqs
        self.times = times
        self.n_epochs = n_epochs
        self.n_tapers = n_tapers
        self.fpath = ''
        self.fname = ''

    def clean(self, con=None):
        """Clean matrix by converting 0 to NaN."""
        if con is None:
            con = self.con
        mat = np.where(self.con == 0, np.nan, self.con)
        return mat

    def threshold(self, con=None, threshold=None):
        """Replace values below threshold with NaNs."""
        if con is None:
            con = self.con
        con = np.abs(con)  # IMPORTANT: Assumes that you want absolute values.
        if threshold is None:
            threshold = np.nanmean(con)
        mat = np.where(self.con < threshold, np.nan, self.con)
        return mat

    def save(self):
        """Save the ConOutput object."""
        if (self.fname or self.fpath) == '':
            print('You must set a filename and path before saving.')
        else:
            filepath = op.join(self.fpath, self.fname)
            with open(filepath, 'wb') as file:
                pickle.dump(self, file)

    def plot(self, con=None, indices=[]):
        """Plot the connectivity output matrix."""
        if con is None:
            con = self.con
        mat = con[:, :, 0]
        if len(indices) > 0:
            mask = np.full(mat.shape, False)
            mask = mask.flatten()
            mask[indices] = True
        plt.matshow(mat[indices])
        plt.colorbar()
        plt.show()

    def plot_thresholded(self, threshold=None):
        """Plot the cleaned, thresholded matrix."""
        mat = self.clean()
        mat = self.threshold(con=mat, threshold=threshold)
        self.plot(con=mat)

    def plot_z(self):
        """Plot the connectivity output matrix as z-scores."""
        mat = self.con[:, :, 0]
        mat = np.abs(mat)
        mat = 0.5 * (np.log(1+mat) - np.log(1-mat))
        plt.matshow(mat)
        plt.colorbar()
        plt.show()

    def plot_roi_connectivity(self, roi, threshold=0.5, parc='HCPMMP1'):
        """Project connectivity measures of an ROI onto a 3d brain."""
        # get 1-dim slice of connections to ROI
        # need some value checks in here...
        roi_con = []
        for i in range(len(self.con)):
            if i < roi:
                roi_con.append(self.con[roi][i][0])
            elif i > roi:
                roi_con.append(self.con[i][roi][0])
            elif i == roi:
                roi_con.append(0)
            else:
                print('error')
        mri_dir = '/Users/joshbear/clinical/meg/anat/'
        brain = Brain('fsaverage', 'split', 'inflated', subjects_dir=mri_dir,
                      cortex='low_contrast', views=['lat', 'med'],
                      background='white', size=(800, 600))
        labels_parc = mne.read_labels_from_annot('fsaverage', parc=parc,
                                                 subjects_dir=mri_dir)
        for idx, label in enumerate(labels_parc):
            if idx == roi:
                label.color = [0, 0, 0, 1]
                brain.add_label(label, borders=False)
            elif idx > 1:
                if np.abs(roi_con[idx]) > np.max(roi_con)*threshold:
                    blue = np.abs(roi_con[idx])*(1/np.max(roi_con))
                    if blue > 1.0:
                        blue = 1.0
                    label.color = [0.1, 0.1, blue, np.abs(roi_con[idx]) *
                                   (1/np.max(roi_con))]
                    brain.add_label(label, borders=False)
        input("Press Enter to continue...")
        return brain

    def plot_roi_conn_z(self, roi, threshold=0.5):
        """Project connectivity measures of an ROI onto a 3d brain."""
        # get 1-dim slice of connections to ROI
        # need some value checks in here...
        roi_con = []
        for i in range(len(self.con)):
            if i < roi:
                roi_con.append(self.con[roi][i][0])
            elif i > roi:
                roi_con.append(self.con[i][roi][0])
            elif i == roi:
                roi_con.append(0)
            else:
                print('error')
        # convert it to a Fisher's Z score
        roi_con = np.abs(roi_con)
        roi_con = 0.5 * (np.log(1+roi_con) - np.log(1-roi_con))
        mri_dir = '/Users/joshbear/clinical/meg/anat/'
        brain = Brain('fsaverage', 'split', 'inflated', subjects_dir=mri_dir,
                      cortex='low_contrast', views=['lat', 'med'],
                      background='white', size=(800, 600))
        labels_parc = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1',
                                                 subjects_dir=mri_dir)
        cmap = mpl.cm.get_cmap('Spectral')
        # cmap.set_over(str(np.max(roi_con)))
        print(f'ROI_CON: {np.max(roi_con)}')
        # cmap.set_under(str(np.max(roi_con) * threshold))
        # cmap.set_under('0.0')
        for idx, label in enumerate(labels_parc):
            if idx == roi:
                label.color = [0, 0, 0, 1]
                brain.add_label(label, borders=False)
            elif idx > 1:
                if np.abs(roi_con[idx]) > np.max(roi_con)*threshold:
                    label.color = cmap(roi_con[idx])
                    brain.add_label(label, borders=False)
                    # print(f'Add {label.name}: value = {roi_con[idx]}')
        fig = plt.figure(figsize=(8, 1))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        # norm = mpl.colors.Normalize(vmin=0, vmax=np.max(roi_con))
        cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                       orientation='horizontal')
        cb.set_label('Absolute Imaginary Coherence')
        plt.show()
        input("Press Enter to continue...")
        return brain
