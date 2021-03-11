"""Module to analyze MEG resting state data using MNE.

This module provides a series of functions and a new data object RestSubject
designed to streamline the processing of global connectivity in resting state
MEG recordings.
"""
import os
import os.path as op
import glob
import mne
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import numpy as np
from surfer import Brain


def get_study_folders(subject):
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
    study_dir = '/Users/joshbear/research/eses_connectivity/'
    subject_dir = op.join(study_dir, 'data', 'subjects')
    meg_dir = op.join(subject_dir, subject, 'ESES')
    mri_dir = '/Users/joshbear/clinical/meg/anat/' + subject
    results_dir = op.join(subject_dir, subject, 'meg', 'results')
    meg_runs = []

    if op.isdir(meg_dir) is False:
        print("MEG data directory (" + meg_dir + ") does not exist.")
        print("Exiting on getUserFolders()")
        return  # make sure calling functions can cope with return = None

    if op.isdir(results_dir) is False:
        print("results_folder (" + results_dir +
              ") does not exist and will be created.")
        os.mkdir(results_dir)

    for file in glob.iglob(meg_dir + '/**/c,rfhp0.1Hz', recursive=True):
        meg_runs.append(file)

    return {'meg': meg_dir, 'anat': mri_dir, 'results': results_dir,
            'runs': sorted(meg_runs)}


def load_subject_data(subject):
    """Given a subject ID, check for a pre-existing data file and load."""
    res_dir = '/Users/joshbear/research/epi_conn/' + subject
    if op.isdir(res_dir) is False:
        # print('Subject <{subject}> does not have a data folder.')
        raise ValueError(f'Subject <{subject}> does not have a data folder.')
    else:
        filepath = f'{res_dir}/{subject}_vars.txt'
        if op.isfile(filepath):
            print(f'Found saved data for <{subject}>.')
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        else:
            raise ValueError(f'Subject <{subject}> does not have saved data.')


def load_conoutput(full_path):
    """Load a saved ConOutput file."""
    if op.isfile(full_path) is False:
        print('Unable to found that object. This requires a full path.')
    else:
        with open(full_path, 'rb') as file:
            return pickle.load(file)


def plot_raw_segment(raw, start=1, stop=10):
    """Pass a raw file and plot, in butterly, the defined segment."""
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False,
                           eog=False, exclude='bads')
    start, stop = raw.time_as_index([start, stop])
    data, times = raw.get_data(picks=picks, start=start, stop=stop,
                               return_times=True)
    for line in data:
        lines = plt.plot(times, line)
        plt.setp(lines, linewidth=0.5)
    plt.show()


def generate_events(raw, duration=60):
    """Generate a list of event marks of defined duration."""
    num_events = raw.n_times / raw.info['sfreq'] / duration
    events = np.zeros(shape=(int(num_events), 3), dtype=int)
    for idx, event in enumerate(events):
        event[0] = idx * raw.info['sfreq'] * duration
        event[1] = 0
        event[2] = 60053
    return events


def get_common_high_corr(con_array, percentile=90):
    """Find ROI-to-ROI connections common across multiple time subsets."""
    mats = [np.abs(mat.con[:, :, 0].flatten()) for mat in con_array]
    locs = []
    for mat in mats:
        adj_percentile = np.percentile(mat[mat > 0], percentile)
        new_locs = np.where(mat > adj_percentile)[0]
        if len(locs) == 0:
            locs = new_locs
        else:
            locs = np.intersect1d(locs, new_locs)
            if len(locs) == 0:
                print(f'Warning: There are no common ROI connections ' +
                      f'at a percentile of {percentile}.')
                return locs
    return locs


class Subject(object):
    """
    A collection of variables for a single-subject resting state scan.

    Args:
        subject (str): Subject ID (used for finding folders)

    Attributes:
        data (str): Directory for MEG data.
        anat (str): Directory for anatomy data.
        res (str) : Directory for output from class functions.
        runs (list) : A list of strings, sorted alphabetically, containing
            full paths to all data files identified on a glob search.
    """

    def __init__(self, subject):
        """Initializer function."""
        self.subject = subject
        self.fname_save = (f'/Users/joshbear/research/epi_conn/{subject}/'
                           f'{subject}_vars.txt')
        if op.isfile(self.fname_save):
            print(f'Saved data for {subject} found. You might want to'
                  f' use load_subject_dat() instead of creating a new'
                  f' instance.')
        self.data = self.get_meg_dir()
        self.anat = self.get_anat_dir()
        self.res, self.res_anat, self.res_data, self.res_pipe, \
            self.res_out = self.get_results_dir()
        self.runs = self.get_all_runs()
        self.mri_dir = '/Users/joshbear/clinical/meg/anat/'  # Root for anat.
        self.meg_dir = '/Users/joshbear/clinical/meg/data/'
        self.imported_runs = []
        self.build_file_structure()
        # Filenames for generated process files.
        self.fname_src = op.join(self.res_anat, '%s-src.fif' % self.subject)
        self.fname_bem = op.join(self.res_anat, '%s-bem.fif' % self.subject)
        self.fname_bem_sol = op.join(self.res_anat, '%s-bem-sol.fif'
                                     % self.subject)
        self.fname_trans = op.join(self.res_pipe, '%s-trans.fif'
                                   % self.subject)
        self.fname_raw = op.join(self.res_data, f'{self.subject}-rest-raw.fif')
        # need: self.fname_cov for the empty room ones (instead of from data)
        self.fname_clean = op.join(self.res_data,
                                   f'{self.subject}-rest-clean.fif')
        self.data_subsets = []  # save custom cropped data files
        self.subsets = []
        # Variables needed for processing...
        self.bads = []  # list of bad channels

    def __repr__(self):
        """Return general information about RestSubject."""
        run_list = ''
        processing_list = self.estimate_completed_steps()
        for idx, run in enumerate(self.runs):
            run_list += '  %i) %s\n' % (idx, run)
        return (f'Subject          : {self.subject}\n'
                f'MEG data folder  : {self.data}\n'
                f'Anatomy folder   : {self.anat}\n'
                f'Results folder   : {self.res}\n'
                f'List of runs     : {len(self.runs)} files found\n'
                f'{run_list}'
                f'Processing steps : (based on presence of output files)\n'
                f'{processing_list}')

    def get_meg_dir(self):
        """Return default MEG directory."""
        return '/Users/joshbear/clinical/meg/data/' + self.subject

    def get_anat_dir(self):
        """Return default MEG directory."""
        return '/Users/joshbear/clinical/meg/anat/' + self.subject

    def get_results_dir(self):
        """Return default results directory."""
        res_dir = '/Users/joshbear/research/epi_conn/' + self.subject
        if op.isdir(res_dir) is False:
            print("Results folder (" + res_dir +
                  ") does not exist and will be created.")
            os.mkdir(res_dir)
        res_anat = res_dir + '/anat'
        if op.isdir(res_anat) is False:
            print('Creating results anatomy folder.')
            os.mkdir(res_anat)
        res_data = res_dir + '/data'
        if op.isdir(res_anat) is False:
            print('Creating results data folder.')
            os.mkdir(res_data)
        res_pipe = res_dir + '/pipeline'
        if op.isdir(res_pipe) is False:
            print('Creating results pipeline data folder.')
            os.mkdir(res_pipe)
        res_out = res_dir + '/output'
        if op.isdir(res_out) is False:
            print('Creating results pipeline data folder.')
            os.mkdir(res_out)
        return res_dir, res_anat, res_data, res_pipe, res_out

    def build_file_structure(self):
        """Create any necessary folders for future analysis."""
        if op.isdir(self.res_anat) is False:
            print("results_folder (" + self.res_anat +
                  ") does not exist and will be created.")
            os.mkdir(self.res_anat)
        if op.isdir(self.res_data) is False:
            print("results_folder (" + self.res_data +
                  ") does not exist and will be created.")
            os.mkdir(self.res_data)
        if op.isdir(self.res_pipe) is False:
            print(f'pipeline results folder {self.res_pipe} does not exist ',
                  f'and will be created.')
            op.mkdir(self.res_pipe)

    def estimate_completed_steps(self):
        """Will estimate completed processing steps based on output files."""
        completed_steps = ''
        order = {'Source space': self.fname_src,
                 'BEM model': self.fname_bem,
                 'BEM solution': self.fname_bem_sol,
                 'Raw MEG import': self.fname_raw,
                 'MEG cleaning': '/placeholder',
                 'Co-registration': self.fname_trans,
                 'Forward model': 'placeholder'}
        for key, val in order.items():
            if op.isfile(val):
                completed_steps += f'  {key:<20}: Completed ({val})\n'
            else:
                completed_steps += f'  {key:<20}: INCOMPLETE\n'
        return completed_steps

    def get_all_runs(self):
        """Return an alphabetically-sorted list of runs in child folders."""
        meg_runs = []
        for file in glob.iglob(self.data + '/**/c,rfhp0.1Hz', recursive=True):
            meg_runs.append(file)
        return sorted(meg_runs)

    def get_run_info(self):
        """Will get some run info. Here for placeholder."""
        # x.runs[0].split('/')[-2]
        return

    def preprocess_anat(self):
        """
        Generate source space and BEM model/solution.

        Will check for source file (-src.fif), BEM file (-bem.fif) and
        BEM solution (-bem-sol.fif). If these files do not exist, they
        will be created and saved in the /anat folder.

        Args:
            None

        Returns:
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
                                         subjects_dir=self.mri_dir)
            mne.write_source_spaces(src_file, src)
        if op.isfile(bem_file):
            print('Found an existing BEM file (%s).' % bem_file)
        else:
            print('Creating BEM using make_bem_model with conductivity '
                  '= 0.3.\nFile will be saved as: %s' % bem_file)
            # mne.bem.make_watershed_bem(x.subject, subjects_dir=x.mri_dir)
            # puts stuff in this directory:
            # /Users/joshbear/clinical/meg/anat/E-0191/bem/watershed
            model = mne.make_bem_model(self.subject, conductivity=[0.3],
                                       subjects_dir=self.mri_dir)
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

    def import_raw(self, resting_state="EPILEPSY24"):
        """
        Import the raw MEG data and perform basic preprocessing steps.

        This will load the raw MEG data, concatenate the resting state runs,
        and save the results (the concatenated resting state and the
        individual files for other types as FIF files.
        """
        imported_runs = []
        resting_runs = []
        for idx, run in enumerate(self.runs):
            if run.split('/')[-4] == 'EPILEPSY24':
                run_type = 'rest'
            elif run.split('/')[-4] == 'RHYTHM':
                run_type = 'sef'
            else:
                run_type = 'other'
            run_number = run.split('/')[-2]
            run_name = run.split('/')[-1]
            run_path = run[0:-len(run_name)]
            run_fname = (f'{self.res_data}/{self.subject}-{run_type}_'
                         f'{run_number:0>2}-raw.fif')
            if op.isfile(run_fname):
                print('%s exists. Checking next run...' % run_fname)
                imported_runs.append(run_fname)
            else:
                raw = mne.io.read_raw_bti(run,
                                          run_path + 'config',
                                          run_path + 'hs_file')
                print('Read %s.\nSaving as %s' % (run, run_fname))
                if run.split('/')[-4] == resting_state:
                    resting_runs.append(raw)
                else:
                    raw.save(run_fname)
                # print('Making noise covariance from full run.')
                # noise_cov = mne.compute_raw_covariance(raw)
                # noise_cov.save(f'{self.res_data}/{self.subject}-{run_type}_'
                #                f'{run_number:0>2}-cov.fif')
                imported_runs.append(run_fname)
        print('All raw data loaded. If you have not done so yet, you should'
              ' create a -trans.fif file (using mne coreg).')
        if len(resting_runs) > 0:
            print(f'There were {len(resting_runs)} run(s) of resting type ',
                  f'({resting_state}) identified and concatenated.')
            raw = mne.concatenate_raws(resting_runs)
            raw.save(f'{self.res_data}/{self.subject}-rest-raw.fif')
        self.imported_runs = imported_runs
        return imported_runs

    def preprocess_raw(self, type='rest', srate=100, fmin=1, fmax=55):
        """Preprocess a raw MEG file and compute the covariance matrix."""
        raw = mne.io.read_raw_fif(self.fname_raw)
        # raw_filt = raw.filter(l_freq=fmin, h_freq=fmax)
        # need to have channel picks
        raw.info['bads'] = self.bads  # alternative location for saving bads
        raw.pick_types(meg=True, ecg=True, exclud='bads')  # ecg has EEG chs
        # For the ICA, it is recommended that we highpass filter ≥1 Hz
        clean = mne.preprocessing.maxwell_filter(raw, st_only=True,
                                                 st_duration=10)

        # Create ICA object and fit it.
        print('Creating ICA object.')
        ica = ICA(n_components=0.95, method='fastica', random_state=0,
                  max_iter=150)
        print('Performing ICA fit.')
        ica.fit(clean, decim=3,
                reject=dict(mag=4e-12, grad=4000e-13),
                verbose='warning')

        # for ECG artifacts
        # ecg_epochs = create_ecg_epochs(clean, ch_name='EEG 001', tmin=-.5,
        #                                tmax=.5, picks=picks)
        print('Calculating ECG-correlated components.')
        ecg_inds, ecg_scores = ica.find_bads_ecg(clean, ch_name='EEG 001',
                                                 method='ctps')

        # for EOG artifacts
        print('Calculating EOG-correlated components.')
        eog_inds, eog_scores = ica.find_bads_eog(clean, ch_name='VEOG',
                                                 threshold=3.0)
        ica.exclude += ecg_inds
        ica.exclude += eog_inds

        print('Applying ICA.')
        ica.apply(clean)
        print('Filtering the bandpass.')
        clean = clean.filter(l_freq=fmin, h_freq=fmax)
        # clean.resample(srate, npad='auto')
        print('Saving the cleaned data.')
        clean.save(self.fname_clean)

        return clean, ecg_inds, ecg_scores, eog_inds, eog_scores
        # need to sace the ica component
        # save the new 'clean' raw file

    def crop_cleaned_data(self, fname, tmin=0, tmax=60):
        """Crop the cleaned data file to the selected time window."""
        try:
            clean = mne.io.read_raw_fif(self.fname_clean)
        except FileNotFoundError:
            print('This subject does not yet have a cleaned data file. ',
                  'You might need to run preprocess_raw() first.')
            return
        clean.crop(tmin, tmax)
        clean.save(f'{self.res_data}/{fname}')
        self.data_subsets.append(fname)

    def generate_subsets(self, type, duration, tstart, tstop):
        """Create a series of cropped data files from the cleaned data."""
        try:
            clean = mne.io.read_raw_fif(self.fname_clean)
        except FileNotFoundError:
            print('This subject does not yet have a cleaned data file. ',
                  'You might need to run preprocess_raw() first.')
            return
        if tstop > len(clean.times) / clean.info['sfreq']:
            tstop = int(len(clean.times) / clean.info['sfreq'])
        window = tstop - tstart
        if window < duration:
            print('The time window given is less than the duration. Exiting.')
            return
        seg_count = window // duration
        segments = []
        print(f'The specified time window of {window} seconds includes '
              f'{seg_count} segment(s) of {duration} seconds each.')
        for i in range(seg_count):
            beg = tstart + (i * duration)
            end = beg + duration
            print(f'Begin at {beg} and end at {end}')
            fname = f'{self.subject}_{type}_{duration}s_t{beg}-{end}_raw.fif'
            if op.isfile(op.join(self.res_data, fname)) is False:
                crop = clean.copy().crop(beg, end - 0.001)
                crop.save(op.join(self.res_data, fname))
            segments.append(f't{beg}-{end}')
        self.subsets.append(RestDataSubset(self.subject, type,
                            duration, segments))
        self.save()

    def preprocess_clean(self, fname=''):
        """Prepare covariance matrix and forward solution for cleaned file."""
        if fname == '':
            print(f'No filename supplied, so will attempt to use the cleaned '
                  f'file: {self.fname_clean}')
            fname = self.fname_clean
        else:
            fname = self.res_data + '/' + fname
        if hasattr(self, 'fname_cov'):
            fname_cov = self.fname_cov
        else:
            fname_cov = fname.split('/')[-1][:-4] + '-cov.fif'
        fname_fwd = fname.split('/')[-1][:-4] + '-fwd.fif'
        try:
            clean = mne.io.read_raw_fif(fname)
        except FileNotFoundError:
            print(f'This subject does not have a data file \'{fname}\'. ',
                  'You might need to run preprocess_raw() first.')
            return
        if op.isfile(op.join(self.res_pipe, fname_cov)):
            print('Covariance matrix file found. Skipping this step.')
        else:
            noise_cov = mne.compute_raw_covariance(clean)
            noise_cov.save(op.join(self.res_pipe, fname_cov))
        if op.isfile(op.join(self.res_pipe, fname_fwd)):
            print('Forward model found. Skipping this step.')
        else:
            src = mne.read_source_spaces(self.fname_src)
            fwd = mne.make_forward_solution(fname,
                                            self.fname_trans,
                                            src,
                                            self.fname_bem_sol,
                                            mindist=5.0,
                                            meg=True,
                                            eeg=False,
                                            n_jobs=1)
            mne.write_forward_solution(op.join(self.res_pipe, fname_fwd), fwd)

    def get_inverse_source(self, fname='', method='dSPM'):
        """Calculate the inverse source estimate."""
        if fname == '':
            print(f'No filename supplied, so will attempt to use the cleaned '
                  f'file as basis: {self.fname_clean}')
            full_path = self.fname_clean
        else:
            full_path = self.res_data + '/' + fname
        if hasattr(self, 'fname_cov'):
            fname_cov = self.fname_cov
        else:
            fname_cov = fname.split('/')[-1][:-4] + '-cov.fif'
        fwd_fname = fname[:-4] + '-fwd.fif'
        snr = 1.0
        lambda2 = 1.0 / snr ** 2
        raw = mne.io.read_raw_fif(full_path)
        fwd = mne.read_forward_solution(op.join(self.res_pipe, fwd_fname))
        cov = mne.read_cov(op.join(self.res_pipe, fname_cov))
        inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,
                                                        depth=None,
                                                        fixed=False)
        stc = mne.minimum_norm.apply_inverse_raw(raw, inv_op, lambda2,
                                                 method)
        return stc

    def get_morphed_source(self, stc):
        """Send inverse estimate and get a version morphed to FSAvg."""
        vertices_to = [np.arange(10242), np.arange(10242)]
        stc_to = mne.morph_data(self.subject, 'fsaverage', stc, n_jobs=1,
                                grade=vertices_to, subjects_dir=self.mri_dir)
        return stc_to

    def get_connectivity_matrix(self, stc, sfreq, fmin=8, fmax=12,
                                mode='mean_flip'):
        """Provide an inverse source estimate and return connectivity."""
        fs_src = mne.read_source_spaces('/Users/joshbear/research/epi_conn/' +
                                        'fsaverage/anat/fsaverage-src.fif')
        labels_parc = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1',
                                                 subjects_dir=self.mri_dir)
        label_ts = stc.extract_label_time_course(labels_parc, fs_src,
                                                 mode=mode,
                                                 allow_empty=True)
        label_ts = [label_ts]
        con = ConOutput(*mne.connectivity.spectral_connectivity(
                        label_ts, method='imcoh', mode='multitaper',
                        sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True,
                        mt_adaptive=True, n_jobs=1))
        return con

    def preprocess_subsets(self, type):
        """Go through each file in the specified subset and preprocess."""
        subset = list(filter(lambda subset: subset.subset_type == type,
                             self.subsets))
        if len(subset) > 1:
            print(f'ERROR: More than one subset with type "{type}" exists.')
            return
        elif len(subset) == 0:
            print(f'ERROR: No subsets of type "type" found.')
            return
        subset = subset[0]
        print(f'Found {len(subset.list_filenames())} file(s).')
        for file in subset.list_filenames():
            self.preprocess_clean(fname=file)
        print('Finished preprocessing subsets.')

    def project_subsets(self, type, fmin=8, fmax=12,
                        mode='mean_flip'):
        """Calculate morphed source estimates for subsets."""
        subset = list(filter(lambda subset: subset.subset_type == type,
                             self.subsets))
        if len(subset) > 1:
            print(f'ERROR: More than one subset with type "{type}" exists.')
            return
        elif len(subset) == 0:
            print(f'ERROR: No subsets of type "type" found.')
            return
        subset = subset[0]
        print(f'Found {len(subset.list_filenames())} file(s).')
        matrices = []
        for file in subset.list_filenames():
            # raw = mne.io.read_raw_fif(op.join(self.res_data, file))
            # sfreq = raw.info['sfreq']
            stc = self.get_inverse_source(file, method='sLORETA')
            stc.save(op.join(self.res_pipe, file[:-8]))
            stc_to = self.get_morphed_source(stc)
            stc_to.save(op.join(self.res_pipe, file[:-7] + 'morphed'))
            # con = self.get_connectivity_matrix(stc_to, sfreq, fmin, fmax,
            # mode)
            # con.fname = file[:-8] + '_con.txt'
            # con.fpath = self.res_out
            # con.save()
            # matrices.append(con)
        return matrices

    def generate_subsets_connectivity(self, type, fmin=8, fmax=12,
                                      mode='mean_flip'):
        """Generate connectivity for a given subset."""
        subset = list(filter(lambda subset: subset.subset_type == type,
                             self.subsets))
        if len(subset) > 1:
            print(f'ERROR: More than one subset with type "{type}" exists.')
            return
        elif len(subset) == 0:
            print(f'ERROR: No subsets of type "type" found.')
            return
        subset = subset[0]
        print(f'Found {len(subset.list_filenames())} file(s).')
        matrices = []
        for file in subset.list_filenames():
            raw = mne.io.read_raw_fif(op.join(self.res_data, file))
            sfreq = raw.info['sfreq']
            stc_to = mne.read_source_estimate(op.join(self.res_pipe,
                                              file[:-7] + 'morphed'))
            con = self.get_connectivity_matrix(stc_to, sfreq, fmin, fmax, mode)
            con.fname = file[:-8] + '_con_' + mode + '.txt'
            con.fpath = self.res_out
            con.save()
            matrices.append(con)
        return matrices

    def load_subsets_connectivity(self, type, mode='mean_flip'):
        """Return a list of connectivity results ConOutput objects."""
        subset = list(filter(lambda subset: subset.subset_type == type,
                             self.subsets))
        if len(subset) > 1:
            print(f'ERROR: More than one subset with type "{type}" exists.')
            return
        elif len(subset) == 0:
            print(f'ERROR: No subsets of type "type" found.')
            return
        subset = subset[0]
        print(f'Found {len(subset.list_filenames())} file(s).')
        matrices = []
        for file in subset.list_filenames():
            con_fname = file[:-8] + '_con_' + mode + '.txt'
            con = load_conoutput(op.join(self.res_out, con_fname))
            matrices.append(con)
        return matrices

    def list_data_subsets(self):
        """Show a list of the data subsets that have been saved."""
        if len(self.data_subsets) == 0:
            print('No data subsets have been saved for this subject.')
        else:
            print('The following data subsets have been saved:')
            for name in self.data_subsets:
                print(f'  - {name}')

    def load_anat_files(self):
        """Load src, bem, and bem-sol."""
        src_file = op.join(self.res_anat, '%s-src.fif' % self.subject)
        bem_file = op.join(self.res_anat, '%s-bem.fif' % self.subject)
        bem_sol_file = op.join(self.res_anat, '%s-bem-sol.fif' % self.subject)
        if not self.src:
            self.src = mne.read_source_spaces(src_file)
        if not self.model:
            self.model = mne.read_bem_surfaces(bem_file)
        if not self.model_solution:
            self.model_solution = mne.read_bem_solution(bem_sol_file)

    def save(self):
        """Save data object using pickle."""
        filepath = f'{self.res}/{self.subject}_vars.txt'
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)


class RestDataSubset(object):
    """
    A container for subsets of resting state data.

    Holds information about a subset that will represent actual files.
    Does not actually contain the data itself.
    """

    def __init__(self, subject, subset_type, duration, segments):
        """Initializer."""
        self.subject = subject
        self.subset_type = subset_type
        self.duration = duration
        self.segments = segments

    def list_filenames(self):
        """Give a list of the associated file names."""
        filenames = []
        for segment in self.segments:
            filename = (f'{self.subject}_{self.subset_type}_{self.duration}s_'
                        f'{segment}_raw.fif')
            filenames.append(filename)
        return filenames


class ConOutput(object):
    """
    A collection of variables for a single-subject resting state scan.

    Args:
        subject (str): Subject ID (used for finding folders)

    Attributes:
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

    def plot_roi_connectivity(self, roi, threshold=0.5):
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
        labels_parc = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1',
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
