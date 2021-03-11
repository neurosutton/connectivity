"""Module to review motion parameters from fMRI data.

This module provides a series of functions for examining and analyzing
the rp_ files output from SPM's realignment procedures.

"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt


def get_motion_parameter_matrix(fname, fpath=False):
    """Import the rp_* data and returns an Nx6 matrix.

    Keyword arguments:
    source -- Location of root directory containing all subjects
    root, file -- One element from results of getParameterFileList

    Returns:
    rp_matrix -- Numpy matrix containing the motion parameters
    """

    if fpath is False:
        full_filename = fname
    else:
        full_filename = op.join(fpath, fname)
    if op.isfile(full_filename) is False:
        print('File could not be found.\n' + full_filename + '\n')
        return False
    else:
        rp_matrix = np.genfromtxt(full_filename)
        return rp_matrix


def get_fd_matrix(fname, fpath=False, radius=50.0):
    """Calculate framewise displacement from an rp_ file.

    Keyword arguments:
    rp_matrix -- Numpy Nx6 matrix containing motion parameters.
    radius    -- For rotational displacement calculations. Default = 50.0 (mm)
    """
    if fpath is False:
        full_filename = fname
    else:
        full_filename = op.join(fpath, fname)
    if op.isfile(full_filename) is False:
        print('File could not be found.\n' + full_filename + '\n')
        return False
    else:
        rp_matrix = np.genfromtxt(full_filename)
    fd_matrix = np.zeros(shape=(rp_matrix.shape[0], 7))
    for i in range(1, len(rp_matrix)):
        for j in range(0, 3):
            fd_matrix[i][j] = rp_matrix[i][j] - rp_matrix[i - 1][j]
        for j in range(3, 6):
            fd_matrix[i][j] = (rp_matrix[i][j] - rp_matrix[i - 1][j]) * radius
        fd_matrix[i][6] = sum(abs(fd_matrix[i]))
    return fd_matrix


def plot_fd(fd_matrix):
    """Plot of motion params and framewise displacements."""
    fd_col_names = ['x', 'y', 'z', 'pitch', 'roll', 'yaw', 'fd']
    plt.plot(fd_matrix)
    plt.legend(fd_col_names)
    for line in plt.gca().lines:
        line.set_linewidth(0.5)  # otherwise, hard to see everything
    plt.show()


def find_best_interval(length, fd_matrix=None, file=None):
    """Identify a series of frames with the lowest displacement.

    Arguments
    ---------
    length          : Length (in slices, not time) of desired interval
    fd_matrix (*)   : Nx7 matrix with the last column being FD values
    file (*)        : Select file for retrieving motion parameters.

    * One of either fd_matrix or file MUST be defined

    Returns
    -------
    index, fd       : index of the start of the best interval
                    : fd is the mean framewise displacement for that interval

    """
    if fd_matrix is None and file is None:
        print('You must either pass a fractional displacement matrix ',
              'or provide the path to a file containing the parameters.')
        return None
    elif file is not None:
        fd_matrix = get_fd_matrix(file)
    index = 0
    fd = np.max(fd_matrix[:, 6])  # guarantees that we will find a lower value
    for i in range(len(fd_matrix) - length + 1):
        new_fd = np.mean(fd_matrix[i:i+length, 6])
        if new_fd < fd:
            index = i
            fd = new_fd
    return index, fd


def find_high_motion_frames(threshold=0.5, fd_matrix=None, file=None,
                            start=0, frames=0):
    """Provide a count of and indices to high motion frames.

    Arguments
    ---------
    threshold       : Movement threshold (in mm framewise displacement)
    fd_matrix (*)   : Nx7 matrix with the last column being FD values
    file (*)        : Select file for retrieving motion parameters.

    * One of either fd_matrix or file MUST be defined

    Returns
    -------
    count, [indices]: index of the start of the best interval
                    : fd is the mean framewise displacement for that interval

    """
    if fd_matrix is None and file is None:
        print('You must either pass a fractional displacement matrix ',
              'or provide the path to a file containing the parameters.')
        return None
    elif file is not None:
        fd_matrix = get_fd_matrix(file)
    if frames == 0:
        frames = len(fd_matrix[:, 6])
    else:
        frames = min(frames + start, len(fd_matrix[:, 6]))
    indices = []
    count = 0
    for idx, fd in enumerate(fd_matrix[start:frames, 6]):
        if fd > threshold:
            indices.append(idx)
            count = count + 1
    return count, indices


def get_motion_details(threshold=0.5, fd_matrix=None, file=None,
                       start=0, frames=0):
    if fd_matrix is None and file is None:
        print('You must either pass a fractional displacement matrix ',
              'or provide the path to a file containing the parameters.')
        return None
    elif file is not None:
        fd_matrix = get_fd_matrix(file)
    if frames == 0:
        frames = len(fd_matrix[:, 6])
    else:
        frames = min(frames + start, len(fd_matrix[:, 6]))
    motion = {}
    motion['indices'] = []
    motion['count'] = 0
    motion['threshold'] = threshold
    motion['file'] = file
    motion['start'] = start
    motion['frames'] = frames
    motion['mean'] = np.mean(fd_matrix[start:frames, 6])
    motion['std'] = np.std(fd_matrix[start:frames, 6])
    for idx, fd in enumerate(fd_matrix[start:frames, 6]):
        if fd > threshold:
            motion['indices'].append(idx + start)
            motion['count'] = motion['count'] + 1
    return motion
