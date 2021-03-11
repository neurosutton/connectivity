"""Process connectivity matrices stored as text files."""

import os.path as op
import os
import csv
import numpy as np

folder = ('/Users/joshbear/research/ied_network/data/subjects/E-0180/' +
          'meg/output/event1_12-30_spike_conn')
conditions = ['a', 'b']  # uses the matrix suffix (just before .) for selection
threshold = None
absolute = True
# sample_file = 'event1_000a.txt'
# new_dir = 'abs_thr80_a_b'

# fullpath = op.join(folder, sample_file)


def read_csv(fullpath):
    with open(fullpath, 'r') as f:
        r = csv.reader(f, delimiter=' ')
        return list(r)


def convert_strings_to_floats(matrix):
    new_matrix = matrix.copy()
    for i in range(len(matrix[0])):
        for j in range(len(matrix[1])):
            new_matrix[i][j] = float(matrix[i][j])
    return new_matrix


def get_absolute_values(matrix):
    new_matrix = matrix.copy()
    return np.abs(new_matrix)


def threshold_matrix(matrix, p):
    cutoff = np.percentile(matrix, p, interpolation='lower')
    matrix[matrix < cutoff] = 0.0
    return matrix


def get_results_directory_name(source_dir, threshold, absolute, conditions):
    target_dir = op.basename(source_dir)
    if conditions is not None:
        target_dir = target_dir + '_' + '_'.join(conditions)
    if absolute is True:
        target_dir = target_dir + '_abs'
    if threshold is not None:
        target_dir = target_dir + '_thr' + str(threshold)
    return op.join(op.dirname(source_dir), target_dir)


def threshold_connectivity_matrix(source_dir, threshold, absolute, conditions):
    new_dir = get_results_directory_name(folder, threshold, absolute,
                                         conditions)
    if op.isdir(new_dir) is False:
        print('Creating results directory: ' + new_dir)
        os.mkdir(new_dir)
    else:
        print('*** ' + new_dir + ' already exists! ***')
        return
    files = os.listdir(source_dir)
    for file in files:
        if file[-4:] == '.txt' and file[-5] in conditions:
            matrix = read_csv(op.join(folder, file))
            matrix = convert_strings_to_floats(matrix)
            if absolute is True:
                matrix = get_absolute_values(matrix)
            if threshold is not None:
                matrix = threshold_matrix(matrix, threshold)
            with open(op.join(new_dir, file), 'w+') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=' ')
                csvWriter.writerows(matrix)


def write_design_matrix(source_dir, conditions):
    file_out = source_dir + '_design_matrix.txt'
    if op.isfile(file_out):
        print('*** ' + file_out + ' already exists! ***')
        return
    files = os.listdir(source_dir)
    files = [file for file in files if file[-5] in conditions]
    with open(file_out, 'a') as f_out:
        for idx in range(int(len(files)/len(conditions))):
            design = np.eye(len(conditions)).astype(int).astype(str)
            for row in design:
                f_out.write(' '.join(row) + '\n')
    f_out.close()


def write_exchange_block_matrix(source_dir, conditions):
    file_out = source_dir + '_exchange_block_matrix.txt'
    if op.isfile(file_out):
        print('*** ' + file_out + ' already exists! ***')
        return
    files = os.listdir(source_dir)
    files = [file for file in files if file[-5] in conditions]
    with open(file_out, 'a') as f_out:
        for idx in range(int(len(files)/len(conditions))):
            f_out.write(str(idx+1) + '\n' + str(idx+1) + '\n')
    f_out.close()


threshold_connectivity_matrix(folder, threshold, absolute, conditions)
write_design_matrix(folder, conditions)
write_exchange_block_matrix(folder, conditions)
