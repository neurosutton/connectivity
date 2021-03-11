"""Object and functions for handling connectivity time series data.

2020.02.11 Copied from IED networks to ESES_connectivity research folder.
"""

import numpy as np
import os.path as op
from os import listdir
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_ind
import pandas as pd
import time
import copy
from bct import nbs_bct as nbs


def load_connectivity_time_series(full_path):
    """Load a saved ConOutput file."""
    if op.isfile(full_path) is False:
        print('Unable to find that object. This requires a full path.')
    else:
        max_bytes = 2 ** 31 - 1
        bytes_in = bytearray(0)
        input_size = op.getsize(full_path)
        with open(full_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        conn_series = pickle.loads(bytes_in)
        return conn_series


def grouped(iterable, n):
    '''
    :param iterable: s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    :param n: how many in each group
    :return: values from iterable in groups of n
    '''
    return zip(*[iter(iterable)]*n)


def progress_report(str, quiet=True):
    if quiet is False:
        print(str)
        return

def get_event_id_from_filename(filename):
    return filename[:-4].split('_')[0]

def get_event_number_from_filename(filename):
    return filename[:-4].split('_')[1]

def get_event_time_from_filename(filename):
    return filename[:-4].split('_')[2]

class ConnectivitySeries(object):

    def __init__(self, output_folder):
        """Initialize function."""
        self.output_folder = output_folder
        # Now, we can easily generate all the important variables...
        self.freq_folders = sorted([d for d in listdir(output_folder) if not d.startswith('.')])
        self.subject = output_folder.split('/')[-4]
        self.freq_files = {}  # will store all files within each frequency bin based on the freq_folders
        self.event_name = output_folder.split('/')[-1].split('_')[0]
        self.event_indices = {}
        # time_range = output_folder.split('/')[-1].split('_')[1].split('-')[1:3]  # will fix when I adjust output_fodler name
        self.time_range = None
        self.time_step = None
        self.time_window = None
        self.data = {}
        self.averaged_data = {}  # Will be generated the first time it is needed.
        self.mask_file = None

    def extract_parameters(self):
        '''Determine time series parameters window, timing, and sliding step length.'''
        param_dict = {}
        for key, value in grouped(self.output_folder.split('/')[-1].split('_'), 2):
            param_dict[key] = value
        self.time_step = float(param_dict['by'])
        self.time_window = float(param_dict['window'])
        negative = False;
        time_start = None;
        time_end = None;
        for item in param_dict['time'].split('-'):
            if time_start == None:
                if item == '':
                    negative = True
                else:
                    try:
                        time_start = int(item)
                        if negative == True:
                            time_start = time_start * -1
                            negative = False
                    except:
                        print('Could not determine start time. Unable to extract parameters.')
                        return
            elif time_end == None:
                if item == '':
                    negative = True
                else:
                    try:
                        time_end = int(item)
                        if negative == True:
                            time_end = time_end * -1
                            negative = False
                    except:
                        print('Could not determine start time. Unable to extract parameters.')
                        return
        self.time_range = [time_start, time_end]

    def load_freq_files(self, quiet=True):
        self.freq_files = {}
        for freq in self.freq_folders:
            progress_report('Loading files for frequency band: ' + str(freq), quiet)
            self.freq_files[freq] = sorted([f for f in listdir(op.join(self.output_folder, freq))
                                            if not f.startswith('.')])
            progress_report('Found ' + str(len(self.freq_files[freq])) + ' files.', quiet)
        return

    def report_freq_files(self):
        if len(self.freq_files) == 0:
            self.load_freq_files()
        summary = {}
        for key in self.freq_files.keys():
            summary[key] = {}
            print('File summary for frequency band: ' + str(key))
            print('- ' + str(len(self.freq_files[key])) + ' files.')
            for file in self.freq_files[key]:
                num = get_event_number_from_filename(file)
                time = get_event_time_from_filename(file)
                if num in summary[key]:
                    summary[key][num] += 1
                else:
                    summary[key][num] = 1
            print('- Total number of events found: ' + str(len(summary[key])))
            print('- Maximum time points found: ' + str(max(summary[key].values())))
            print('- Minimum time points found: ' + str(min(summary[key].values())))

    def load_data_for_band(self, freq, quiet=True, reload=False):
        if freq not in self.freq_folders:
            print('ERROR: No folder for frequency band ' + freq)
            return
        if len(self.freq_files) == 0:
            self.load_freq_files()
        if freq in self.data.keys() and reload is False:
            progress_report('EXITING: Data has already been loaded for frequency band ' + freq, quiet)
            return
        else:
            start = time.time()
            self.data[freq] = []
            for file in [f for f in self.freq_files[freq] if f.startswith(self.event_name)]:
                num_and_time = file[:-4].split('_')[1:3]
                event_number = get_event_number_from_filename(file)
                event_time = get_event_time_from_filename(file)
                if not event_number in self.event_indices:
                    self.event_indices[event_number] = len(self.event_indices)
                if self.event_indices[event_number] >= len(self.data[freq]):
                    self.data[freq].append([])
                matrix = pd.read_csv(op.join(self.output_folder, freq, file), sep=' ', header=None)
                matrix = matrix.to_numpy()
                progress_report(freq + ' | Event ' + event_number + ' at time ' + event_time, quiet)
                self.data[freq][self.event_indices[event_number]].append(matrix)
            end = time.time()
            progress_report('Loading data took ' + str(end - start) + ' seconds.', quiet)

    def load_data(self, quiet=True, reload=False):
        # Collect all files by frequency bin
        if len(self.freq_files) == 0:
            self.load_freq_files()
        if self.time_window is None:
            self.extract_parameters()
        # Now loop through each frequency band and load the data.
        for freq in self.freq_folders:
            self.load_data_for_band(freq, quiet=quiet, reload=reload)

    def save(self):
        """Save the ConnectivitySeries object."""
        filename = self.event_name + '_times_series_data.txt'
        filepath = op.join(self.output_folder, filename)
        max_bytes = 2 ** 31 - 1
        bytes_out = pickle.dumps(self)
        with open(filepath, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx + max_bytes])

    def show_frequency_bands(self):
        print('Data are available in the following frequency bands:')
        for idx, f in enumerate(self.freq_folders):
            print(f'{idx}) {f}')

    def show_start_times(self, freq_band=0):
        if type(freq_band) is int:
            freq_band = self.freq_folders[freq_band]
        if freq_band not in self.data:
            print('Data for ' + freq_band + ' not loaded. Will attempt to load...')
            self.load_data_for_band(freq=freq_band)
        print('Data in the frequency band ' + freq_band + ' are available for the following start times:')
        print(np.arange(self.time_range[0], self.time_range[1],
                        self.time_step)[0:np.ma.size(self.data[freq_band], axis=1)])

    def get_connectivity_matrix_at_time(self, freq, time):
        if freq not in self.freq_folders:
            print('ERROR: ' + str(freq) + ' is not available in this data set.')
        all_times = np.arange(self.time_range[0], self.time_range[1],
                              self.time_step)[0:np.ma.size(self.data[freq], axis=1)]
        time_index = np.where(all_times == time)[0][0]
        matrix = np.swapaxes(self.data[freq].copy(), 0, 1)[time_index]
        return matrix

    def calculate_nbs(self, freq, time1, time2, thresh, paired=False, k=1000, tail='both', save_path=None):
        matrix1 = np.absolute(self.get_connectivity_matrix_at_time(freq, time1))
        matrix2 = np.absolute(self.get_connectivity_matrix_at_time(freq, time2))
        matrix1 = np.swapaxes(matrix1, 0, 2)
        matrix2 = np.swapaxes(matrix2, 0, 2)
        nbsoutput = nbs(matrix1, matrix2, thresh=thresh, paired=paired, k=k, tail=tail)
        if save_path is not None:
            if op.isdir(save_path):
                print('Trying to save to ' + save_path)
                if tail == 'left':
                    sign = '<'
                elif tail == 'right':
                    sign = '>'
                else:
                    sign = '<>'
                save_name = (self.event_name + '_nbs_w' + str(self.time_window) + '_ts' + str(self.time_step) + '_' +
                             freq + '_' + str(time1) + sign + str(time2) + '_at_t>' + str(thresh))
                print(save_name)
                int_output = copy.deepcopy(nbsoutput[1]).astype(bool).astype(int)
                np.savetxt(op.join(save_path, save_name + '.txt'), int_output, fmt='%i')
                np.savetxt(op.join(save_path, save_name + '_averaged_t' + str(time1) + '.txt'),
                           np.average(matrix1, axis=2))
                np.savetxt(op.join(save_path, save_name + '_averaged_t' + str(time2) + '.txt'),
                           np.average(matrix2, axis=2))
            else:
                print('ERROR: save_as must be a valid file path.')
        return nbsoutput

    def get_baseline_mean(self, baseline, con_values):
        """Obtain a baseline connectivity value from baseline seconds at the start."""
        num_values = int(baseline / self.time_step - (self.time_window / self.time_step) + 1)
        return np.average(con_values[0:num_values])

    def get_time_series_mean(self, freq_band, mask=False):
        ts_mean = copy.deepcopy(self.data[freq_band])
        if mask is not False:
            if type(mask) is bool:
                nodes = np.array(np.loadtxt(self.mask_file), dtype=bool)
            elif type(mask) is np.ndarray:
                nodes = mask.astype(bool)
            nodes = nodes.astype(float)
            nodes[nodes <= 0] = np.nan
            for i, spike in enumerate(ts_mean):
                for j, time in enumerate(spike):
                    ts_mean[i][j] = ts_mean[i][j] * nodes
        ts_mean = np.nanmean(np.absolute(ts_mean), axis=(2, 3))
        return ts_mean

    def calculate_averaged_data(self, freq_band):
        if freq_band not in self.data:
            print('ERROR: No such frequency band in the original data.')
        else:
            self.averaged_data[freq_band] = np.average(np.absolute(self.data[freq_band][:].copy()), axis=0)

    def plot_connectivity(self, freq_band=0, mask=False, baseline=None, sig=0.05):
        """
        Plot the connectivity values over time.
        :param freq_band: Which frequency band to use (str)
        :param mask: file if using a mask (should change to mask)
        :param baseline: (int), in seconds, to use as baseline value for significance testing
        :param sig: (float), what level is considered significant from baseline
        :return:
        """
        print('Entered plot_connectivity function.')
        if self.time_range is None:  # Make sure the time data have been loaded.
            self.extract_parameters()
        if type(freq_band) is int:
            freq_band = self.freq_folders[freq_band]
        if freq_band not in self.data:
            self.load_data_for_band(freq_band)
        if freq_band not in self.averaged_data:
            self.calculate_averaged_data(freq_band)
        if mask is not False:
            if mask is True:
                nodes = np.array(np.loadtxt(self.mask_file), dtype=bool)
            elif type(mask) is np.ndarray:
                nodes = mask.astype(bool)
            else:
                print('ERROR: Mask parameter is problematic. Exiting.')
                return
            masked_data = [m[nodes] for m in self.averaged_data[freq_band].copy()]
            con_values = np.average(masked_data, axis=1)
            plt.plot(con_values)
            plt.title('Total Within-Network Connectivity, ' + freq_band + ' Hz')
        else:
            title = f'Whole Brain Connectivity, {freq_band} Hz'
            con_values = np.average(self.averaged_data[freq_band].copy(), axis=(1, 2))
            plt.plot(con_values)
            plt.title(title)
        offset = int(float(self.time_window) / float(self.time_step))
        xlabels = list(range(self.time_range[0], self.time_range[1] + 1, 1))
        xlocs = list(range(-offset,
                           np.shape(self.data[freq_band])[1] + int(1.0 / float(self.time_step)) - offset,
                           int(1.0 / float(self.time_step))))
        plt.xlabel('Time (in Seconds)')
        plt.ylabel('| Imaginary Part of Coherence |')
        locs, labels = plt.xticks()
        plt.xticks(xlocs, xlabels)
        plt.axvline(x=xlocs[int(np.floor(len(xlocs)/2))], linestyle='--', dashes=(5, 2), color='m', alpha=0.25)
        if baseline is not None:
            try:
                baseline_mean = self.get_baseline_mean(baseline, con_values)
                plt.axhline(y=baseline_mean, alpha=0.25)
                # should add some code to identify the range used for the baseline data
                ts_mean = self.get_time_series_mean(freq_band=freq_band, mask=mask)
                ts_sig = np.mean(ts_mean, axis=0)
                num_values = int(baseline / self.time_step - (self.time_window / self.time_step) + 1)
                for idx, time in enumerate(ts_sig):
                    if ttest_ind(ts_mean[:, 0:num_values].flatten(), ts_mean[:, idx])[1] > sig:
                        ts_sig[idx] = np.nan
                plt.plot(ts_sig, 'kx')
            except:
                print('A baseline was requested, but the value of ' + str(baseline) + ' could not be used.')
        plt.show()

    def plot_connectivity_matrix(self, freq_band=0, time=0, vmin=None, vmax=None, title=None, cmap='jet'):
        if type(freq_band) is int:
            freq_band = self.freq_folders[freq_band]
        if len(self.averaged_data.keys()) == 0:
            for key in self.data.keys():
                self.averaged_data[key] = np.average(np.absolute(self.data[key][:].copy()), axis=0)
        all_times = np.arange(self.time_range[0], self.time_range[1],
                              self.time_step)[0:np.ma.size(self.data[freq_band], axis=1)]
        time_index = np.where(all_times == time)[0][0]
        plt.matshow(self.averaged_data[freq_band][time_index], cmap=cmap, vmin=vmin, vmax=vmax)
        if title is not None:
            plt.title(title)
        cbar = plt.colorbar()
        cbar.set_label('| Imaginary Part of Coherence |')
        plt.show()

    def write_conn_matrix(self, fname_output, freq_band=0, time=0):
        if type(freq_band) is int:
            freq_band = self.freq_folders[freq_band]
        if len(self.averaged_data.keys()) == 0:
            for key in self.data.keys():
                self.averaged_data[key] = np.average(np.absolute(self.data[key][:].copy()), axis=0)
        matrix = self.averaged_data[freq_band][time].copy()
        np.savetxt(fname_output, matrix)

"""Some key code steps for next implementation:

To show a connectivity matrix from the data:


Probably worth saving processing time by averaging across the spikes early:
avedata = np.average(np.abs(data['12-30'][:]), axis=0)

To plot a time course for any two ROIs (example 34, 120):
plt.plot(avedata[:,43, 106], label='43, 106')
plt.plot(avedata[:,115, 106], label='115, 106')
plt.legend()
plt.show()

Getting the NBS network and using it:
nbs = np.loadtxt('/Users/joshbear/research/ied_network/data/subjects/E-0180/meg/output/nbs/pre<spike_freq_12_25_t_3.35_extent.txt')
mask = np.array(nbs, dtype=bool)
avedata[0][mask] (GIVES TWICE AS MANY AS THERE SHOULD BE, BUT WON'T MATTER FOR AVERAGING...)
np.average(avedata[22][mask]) (GIVES THE AVERAGE FOR THAT TIME POINT)

This can create an array of the masked arrays:
avedata = np.average(np.abs(data['12-30'][:]), axis=0)
avs = [m[mask] for m in avedata]
plt.plot(np.average(avs, axis=1))
plt.title('Total Within-Network Connectivity, 12-30Hz')
plt.show()

"""

"""
# Making a list of ROIs in the mask:
nbs = np.loadtxt('/Users/joshbear/research/ied_network/data/subjects/E-0180/meg/output/nbs/pre<spike_freq_12_25_t_3.35_extent.txt')
mask = np.array(nbs, dtype=bool)
masked_rois = []
for idx, row in enumerate(mask):
    if True in row:
        masked_rois.append(idx)

print(masked_rois)

# Create a new mask with every connection to a masked ROI set to False
mask2 = mask.copy()
for idx, row in enumerate(mask2):
    if idx in masked_rois:
        row.fill(True)
    else:
        row[masked_rois] = True

# Then, the new mask can be used to average out all values in the network, or outside the network, to the whole brain.
avs = [m[~mask2] for m in avedata]
plt.plot(np.average(avs, axis=1))
plt.title('Total Outside-Network Connectivity, 12-30Hz')
plt.show()

avs = [m[mask2] for m in avedata]
plt.plot(np.average(avs, axis=1))
plt.title('Total Outside-Network Connectivity, 12-30Hz')
plt.show()



import connectivity_time_series as cts
nbs = '/Users/joshbear/research/ied_network/data/subjects/E-0180/meg/output/nbs/pre<spike_freq_12_25_t_3.35_extent.txt'
m = cts.load_connectivity_time_series('/Users/joshbear/research/ied_network/data/subjects/E-0215/meg/output/event2_-10-10_by_0.5_con/event2_times_series_data.txt')
import numpy as np
import matplotlib.pyplot as plt
nodes = np.array(np.loadtxt(nbs), dtype=bool)
masked_data = [m[nodes] for m in m.averaged_data['12-30']]
all_data = [m[nodes_all] for m in m.averaged_data['12-30']]

xlabels = list(range(-5, 6, 1))
xlocs = list(range(-1, 43, 4))
plt.plot(np.average(masked_data, axis=1))
plt.title('Mean SAN Connectivity, 12–30 Hz')
plt.xlabel('Time (in Seconds)')
plt.ylabel('| Imaginary Part of Coherence |')
locs, labels = plt.xticks()
plt.xticks(xlocs, xlabels)
plt.axvline(x=19, linestyle='--', dashes=(5,2), color='m')

plt.show()

plt.plot(np.average(all_data, axis=1))
plot.show()




FOR MY 4-DIMENSIONAL DATA STRUCUTRE, CTS.DATA[12-30], THIS GIVES ME THE AVERAGE AT EACH TIME POINT:
np.average(np.average(d, axis=0), axis=(1,2))
(assuming the data have already been changed to absolute values)



def time_pandas():
   start = time.time()
   mp = pd.read_csv(file, sep=' ', header=None)
   mp = mp.to_numpy()
   end = time.time()
   print(end - start)


def time_numpy():
    start = time.time()
    m = np.loadtxt(file)
    end = time.time()
    print(end - start)



np.shape(c.data['12-30'])
np.shape(c.data['30-55'])
np.sum(c.data['12-30'])
np.sum(c.data['30-55'])
np.sum(c.averaged_data['12-30'])
np.sum(c.averaged_data['30-55'])


"""
