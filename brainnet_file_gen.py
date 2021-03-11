"""Collection of tools to generate relevant network files for visualization in BrainNet Viewer

2020.02.13 Copied to eses_connectivity folder for any relevant modifications. 
"""
import os.path as op
from os import listdir
import numpy as np

# mask = np.array(nbs, dtype=bool)

def get_nbs_mask(file):
    network_data = np.loadtxt(file)
    mask = np.array(network_data, dtype=int)
    return mask

def get_parcellation_labels(label_file):
    m_in = open(label_file, 'r')
    node_labels = m_in.readlines()
    for idx, label in enumerate(node_labels):
        node_labels[idx] = label.strip()
    return node_labels

def get_parcellation_coords(coordinate_file):
    m_in = open(coordinate_file, 'r')
    node_coords = m_in.readlines()
    for idx, coord in enumerate(node_coords):
        node_coords[idx] = coord.strip()
    return node_coords

def get_nbs_nodes(file):
    network_data = np.loadtxt(file)
    mask = np.array(network_data, dtype=bool)
    masked_rois = []
    for idx, row in enumerate(mask):
        if True in row:
            masked_rois.append(idx)
    return masked_rois

def validate_data(labels, coords, network_nodes, quiet=True):
    if quiet is False:
        print(f'# Labels: {len(labels)}\n# Coords: {len(coords)}\nNodes of interest: {network_nodes}')
    node_oob = False
    for node in network_nodes:
        if node < 0:
            node_oob = True
        elif node > len(labels):
            node_oob = True
    if len(labels) != len(coords):
        raise ValueError('Length of label and coordinate files are not equal.')
    if node_oob is True:
        raise IndexError('The list of network nodes includes invalid indices.')
    return

def write_brainnet_nodes(filename, labels, coords, network_nodes, title=None, conn_file=None):
    """ Create a .node file for use in BrainNet Viewer.
    :param filename: .node path and file to be written
    :param labels:
    :param coords:
    :param network_nodes:
    :param title:
    :param conn_file: If given, will assign connectivity values to the nodes based on their strongest connection.
    :return:
    """
    try:
        validate_data(labels, coords, network_nodes, quiet=False)
    except:
        print('Unable to validate the labels, coordinates, and network nodes provided.')
    if title is None:
        title = 'No Title Given'
    if conn_file is not None:
        conn = np.array(np.loadtxt(conn_file), dtype=float)
    m_out = open(filename, 'w')
    m_out.write('#' + title + '\n')
    for idx, coord in enumerate(coords):
        if conn_file is not None:
            if idx in network_nodes:
                color = 2.0
                value = np.max(conn[idx][:])
            else:
                color = 1.0
                value = 0.0
        elif idx in network_nodes:
            color = 2.0
            value = 3.0
        else:
            color = 1.0
            value = 1.0
        label_name = labels[idx].lstrip('L|R_').rstrip('_ROI-l|rh')
        line_out = f'{coord} {color} {value} {label_name}\n'
        m_out.write(line_out)
    m_out.close()

def write_brainnet_edges(fname_conn, fname_output, fname_nbs):
    """Create a .edge file for use in BrainNet Viewer.

    INPUTS
    fname_conn: A connectivity matrix, generally a .txt file.
    fname_output: Where to write the output.
    fname_nbs: The NBS network of interest.
    """
    conn = np.array(np.loadtxt(fname_conn), dtype=float)
    mask = np.array(np.loadtxt(fname_nbs), dtype=int)
    edges = conn*mask
    np.savetxt(fname_output, edges)

def get_sorted_text_files(output_dir, print_output=True):
    files = sorted([f for f in listdir(output_dir) if f.endswith('.txt')])
    if print_output is True:
        print('The files are: ')
        for idx, file in enumerate(files):
            print(str(idx).rjust(2) + ': ' + file)
    else:
        return files

def get_edge_filename(fname_nbs):
    return fname_nbs[:-3] + 'edge'

def get_node_filename(fname_nbs):
    return fname_nbs[:-3] + 'node'

def generate_brainnet_files(output_dir, fname_nbs, fname_conn):
    files = get_sorted_text_files(output_dir, print_output=False)
    if type(fname_nbs) is int:
        fname_nbs = files[fname_nbs]
    if type(fname_conn) is int:
        fname_conn = files[fname_conn]
    if not op.isfile(op.join(output_dir, fname_nbs)):
        print('ERROR: ' + op.join(output_dir, fname_nbs) + ' does not exist.')
        return
    label_file = '/Users/joshbear/research/ied_network/scripts/HCPMMP1-labels.txt'
    coordinate_file = '/Users/joshbear/research/ied_network/scripts/HCPMMP1-coords.txt'
    fname_edges = op.join(output_dir, get_edge_filename(fname_conn))
    fname_nodes = op.join(output_dir, get_node_filename(fname_conn))
    labels = get_parcellation_labels(label_file)
    coords = get_parcellation_coords(coordinate_file)
    network_nodes = get_nbs_nodes(op.join(output_dir, fname_nbs))
    write_brainnet_nodes(fname_nodes, labels, coords, network_nodes, conn_file=op.join(output_dir, fname_conn))
    write_brainnet_edges(op.join(output_dir, fname_conn), fname_edges, op.join(output_dir, fname_nbs))
    print('The following two files were generated:')
    print(' ' + fname_edges)
    print(' ' + fname_nodes)

"""
import brainnet_file_gen as bg

label_file = '/Users/joshbear/research/ied_network/scripts/HCPMMP1-labels.txt'
coordinate_file = '/Users/joshbear/research/ied_network/scripts/HCPMMP1-coords.txt'
fname_nbs = '/Users/joshbear/research/ied_network/data/subjects/ESES_30/meg/output/event1_nbs_w0.5_ts0.25_12-30_-5<0_at_t>4.txt'
fname_conn = '/Users/joshbear/research/ied_network/data/subjects/ESES_30/meg/output/event1_nbs_w0.5_ts0.25_12-30_-5<0_at_t>4_averaged_t0.txt'
fname_edges = '/Users/joshbear/research/ied_network/data/subjects/ESES_30/meg/output/event1_nbs_w0.5_ts0.25_12-30_-5<0_at_t>4.edge'
fname_nodes = '/Users/joshbear/research/ied_network/data/subjects/ESES_30/meg/output/event1_nbs_w0.5_ts0.25_12-30_-5<0_at_t>4.node'


labels = bg.get_parcellation_labels(label_file)
coords = bg.get_parcellation_coords(coordinate_file)
network_nodes = bg.get_nbs_nodes(fname_nbs)
bg.write_brainnet_nodes(fname_nodes, labels, coords, network_nodes, conn_file=fname_conn)
bg.write_brainnet_edges(fname_conn, fname_edges, fname_nbs)
print(network_nodes)
"""
