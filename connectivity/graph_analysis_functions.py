import shared
from tqdm import tqdm
import nia_stats_and_summaries as nss
import networkx as nx
import numpy as np
from networkx.algorithms import community
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import networkx.algorithms.community as nx_comm
from sklearn import metrics
from random import shuffle
from networkx.algorithms import community

# BMS
import os
from math import ceil
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fmri_analysis_get_data as get
import fmri_analysis_manipulations as fam
import fmri_analysis_bnv_prep as bnv_prep
import fmri_analysis_utilities as utils
utils.check_data_loaded()
# >>>END BMS

# CREATE A FUNCTION TO SIMPLIFY PLOTTING and GRAPH MEASURES


def plot_weighted_graph(
        gw,
        network=None,
        color_nodes_by=None,
        cmap=None,
        node_cmap=None,
        node_size=300,
        **kwargs):
    """Plot the edges and nodes in the selected graph.

       Should look like an axial glass view, if the orthocenter coordinates
       for the atlas nodes are passed by way of pos kwarg.

       Parameters
       ----------
       gw : nx.Graph
           Graph from full-rank connectivity matrix. (May not have to
           be from the full-rank results.)
       color_nodes_by : string or function
           If string identifier of a parameter/feature given to
               the nodes in gw. Typically this will be assigned with
               the function nx.set_node_attributes(g, dict) by passing a
               dictionary of node / parameter / value pairs,
               in form of {node: {'param': value}}
           If function, must accept gw as the only parameter and must return
               a dict of node:value pairs. Works with, for example, NX's clustering()
       **kwargs : Optional parameters. Presently accepts the following:
       pos (kwarg) : dict
           Dictionary with coordinates for each node. Possibly
           extensible with other properties in pos. See nx
       node_weights (kwarg) : list
           The list should include a label and the corresponding nx
           property. e.g., ['degree', gw.degree]

       Returns
       -------
       Nothing, displays graph with edges, color-coordinated depending
       on kwargs.

       TO DO
       -----
       1. Marked below as well. Not sure if the node_weights argument
          will properly assign the values of the weight or if they
          will simply color by the node ID. Ran into this problem
          in the color_nodes_by section above, fixed by ensuring I was
          using the dict values instead of dict keys (JJB)
    """
    gc = gw.copy()

    # Find min/max edge weights
    v = _find_colorbar_limits([d['weight']
                               for (u, v, d) in gc.edges(data=True)])

    options = {
        "width": 1.5,
        "node_size": node_size,
        "linewidths": 1,
        "edgecolors": 'black',
        "node_color": 'yellow',
        "edge_cmap": cmap,
        "edge_vmin": -v,
        "edge_vmax": v,
        "with_labels": False,
        "vmin": -0.1,
        "vmax": 0.1
    }
    if network is not None:
        pos = get_position_dict(network)
        options['pos'] = pos
        # nodes = gw.nodes()
        for node in list(gc):
            if node not in pos.keys():
                gc.remove_node(node)
    elif 'pos' in kwargs.keys():
        options['pos'] = kwargs['pos']

    # Create an ordered list of the edges
    # The key for both options will sort the values based on the absolute value
    # of the edge weights. This allows highly negative values (favoring the second
    # comparison group in a difference map) to be plotted on top alongside strong
    # positive connections
    options['edgelist'] = sorted(
        gc.edges(
            data=True), key=lambda t: abs(
            t[2].get(
                'weight', 1)))
    eweights = [d['weight'] for (u, v, d) in gc.edges(data=True)]
    options['edge_color'] = sorted(eweights, key=abs)

    cmap = plt.cm.hsv if cmap is None else cmap
    options['cmap'] = cmap

    if color_nodes_by is not None:
        if callable(color_nodes_by):
            try:
                color_return = dict(color_nodes_by(gc))
                color_dict = {}
                for key in color_return:
                    color_dict[key] = {'func_color': color_return[key]}
                nx.set_node_attributes(gc, color_dict)
                color_nodes_by = 'func_color'
            except TypeError:
                print('color_nodes_by parameter could not be used')

        if isinstance(color_nodes_by, str):
            options['node_color'] = [
                [v for v in nx.get_node_attributes(gc, color_nodes_by).values()]]

        node_v = _find_colorbar_limits(options['node_color'][0])
        options['vmin'] = -node_v
        options['vmax'] = node_v

    elif 'node_weights' in kwargs.keys():
        add_node_weights(
            gc,
            kwargs['node_weights'][0],
            kwargs['node_weights'][1])
        # TO DO: double check this next bit isn't just taking the node keys
        options['node_color'] = [
            v for v in nx.get_node_attributes(
                gc, kwargs['node_weights'][0])]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw(gc, ax=ax, **options)
    norm = mpl.colors.Normalize(vmin=-v, vmax=v, clip=False)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.ax.set_ylabel('Edges')

    if color_nodes_by:
        # Option for second colorbar based on node weight or other
        # characteristic
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("bottom", size="5%", pad=0.2)
        node_norm = mpl.colors.Normalize(vmin=options['vmin'],
                                         vmax=options['vmax'],
                                         clip=False)
        cmap = cmap if not node_cmap else node_cmap
        fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=node_norm,
                cmap=cmap),
            cax=ax_cb,
            ax='vertical',
            orientation='horizontal')
        ax_cb.set_xlabel('Nodes')

    plt.show()


def _find_colorbar_limits(vals):
    if np.absolute(max(vals)) > np.absolute(min(vals)):
        return (.8 * ceil(np.absolute(max(vals))))
    else:
        return (.8 * ceil(np.absolute(min(vals))))


def print_graph_measures(gw):
    print(f'{nx.average_shortest_path_length(gw)}')
    print(f'{nx.average_shortest_path_length(gw, weight="weight")}')
    print(f'{nx.average_clustering(gw)}')
    print(f'{nx.average_clustering(gw, weight="weight")}')
    return


def make_graph_without_nans(matrix):
    """
    Input matrix is original matrix that may be masked for a given network and/or person
    matrix = get.get_network_matrix(network_name, subj_id,
                                            network_mask,
                                            conn_data=subj_data)
    Note: network_mask is for thresholded networks. This thresholded network is appropriate when the proportional threshold is based on population-level analysis. Otherwise, individual proportional thresholds can be invoked through get_network_matrix with the keyword "prop_thr" and no "network_mask"
    """
    graph = nx.Graph(matrix)
    edges_to_remove = []
    for edge in graph.edges(data=True):
        if np.isnan(edge[2]['weight']):
            edges_to_remove.append((edge[0], edge[1]))
    graph.remove_edges_from(edges_to_remove)
    return graph


def largest_subgraph(g):
    gc = []
    for val in nx.connected_components(g):
        gc.append(val)
    # gc = list(nx.connected_components(g))
    max_set = max(gc, key=len)
    return g.subgraph(max_set).copy()


def find_auc_for_measure(measure, df):
    # measure = 'shortest_path_length'
    subjects = sorted(set(df['subject']))
    auc_df = pd.DataFrame(columns=['subject', 'group', 'auc'])
    thresholds = sorted(set(df['threshold']))
    for subject in subjects:
        idx = len(auc_df)
        auc = metrics.auc(thresholds, df[df['subject'] == subject][measure])
        auc_df.at[idx, 'subject'] = subject
        auc_df.at[idx, 'auc'] = auc
        auc_df.at[idx, 'group'] = df[df['subject'] == subject]['group'].tolist()[
            0]
    return auc_df


def mean_auc_by_group(df):
    results = {}
    for group in set(df['group']):
        results[group] = np.mean(df[df['group'] == group]['auc'])
    return results


def randomize_group_assignments(df):
    new_df = df.copy()
    random_groups = new_df['group'].tolist()
    shuffle(random_groups)
    new_df['group'] = random_groups
    return new_df


# a little function to get a graph with certain subjects dropped
def drop_subjects_from_df(df, subjects):
    new_df = df.copy()
    for subject in subjects:
        new_df = new_df[new_df['subject'] != subject]
    new_df['average_clustering'] = pd.to_numeric(
        new_df['average_clustering'], downcast='float')
    new_df['shortest_path_length'] = pd.to_numeric(
        new_df['shortest_path_length'], downcast='float')
    new_df['global_efficiency'] = pd.to_numeric(
        new_df['global_efficiency'], downcast='float')
    new_df['mean_degree'] = pd.to_numeric(
        new_df['mean_degree'], downcast='float')
    new_df['Q'] = pd.to_numeric(new_df['Q'], downcast='float')
    return new_df


def compare_measures(graph_df, measure):
    return ttest_ind(graph_df[graph_df['group'] == 'hc'][measure],
                     graph_df[graph_df['group'] == 'eses'][measure])


# >>>BMS
class current_analysis():
    """Local class to pass some preset fields to the parallelization
       functions."""

    def __init__(
            self,
            grouping_col='group',
            prop_thr=None,
            subgraph_network=None):
        self.grouping_col = grouping_col
        self.prop_thr = prop_thr
        self.subgraph_network = subgraph_network


def get_position_dict(network):
    """Add position information to each node using the coordinates in a dataframe with ROI labels and coordinates.
    Use with plot_weighted_graphs to represent the results in an axial super glass brain.
    Currently, not flexible as the specific label column is hard coded."""
    bnv = bnv_prep.bnv_analysis(network=network, prop_thr=None)
    network_locs = bnv.limit_labels(network=network)
    position_dict = {}

    for index, row in network_locs.iterrows():
        position_dict[index] = [row['x'], row['y']]

    return position_dict


'''
def get_position_dict(network):
    """Add position information to each node using the coordinates in a dataframe with ROI labels and coordinates.
    Use with plot_weighted_graphs to represent the results in an axial super glass brain.
    Currently, not flexible as the specific label column is hard coded."""
    bnv = bnv_prep.bnv_analysis(network=network, prop_thr=None)
    network_locs = bnv.limit_labels(network=network)
    roi_dict = OrderedDict()
    for n, net in enumerate(network_locs['SuttonLabel']):
        roi_dict[n] = network_locs.loc[network_locs['SuttonLabel']
                                       == net, ['x', 'y']].values[0]
    position_dict = {}
    for k, v in roi_dict.items():
        position_dict[k] = v

    return position_dict
'''


def add_node_weights(G, msr_name, nx_func):
    """
    Helper function to add weights (values) and a dictionary key
    to the Graph.node structure.

    Parameters
    ----------
    G = graph
    msr_name = key entry for nodal weighting
    nx_func = NetworkX function to generate the weights per node

    Returns
    -------
    Modifies the Graph node dictionary to include the new weighting
    values and key.
    """

    msr_dict = nx_func
    for n in G.nodes:
        G.nodes[n][msr_name] = msr_dict[n]


def sort_edge_weights(G, verbose=False):
    """Helper function. Extract the weights, sort them, and find the matching
    values for a sorted list. Needed for percentile thresholding to supplement
    the MST selection.

    Returns
    -------
    sorted edges : list
        edges sorted by ascending weight
    sorted weights : list
        ascending edge weights
    """

    weights_dict = {}
    for u, v, weight in G.edges.data("weight"):
        weights_dict[weight] = [u, v]

    sorted_weights = sorted(list(weights_dict.keys()))
    if sorted_weights:
        sorted_edges = sorted([])
        for wt in sorted_weights:
            sorted_edges.append(weights_dict[wt])
        if verbose:
            print(
                f'Sorted edge weights = beginning {sorted_weights[0]} and end {sorted_weights[-1]}')
            print(f'pop() defaults to the end of the list.')
        return sorted_edges, sorted_weights
    else:
        raise ValueError('Graph did not have sortable weights.\n')


def add_thr_edges(G, prop_thr=None, verbose=False):
    """
    Computes MST for whole brain and then adds subset of edges back
    to the MST graph, depending on proportional threshold for highest
    weighted edges.

    Parameters
    ----------
    G : nx.Graph
    prop_thr : float
    proportional threshold (optional) for adding nodes to
        the MST result.

    Returns
    -------
    thresholded_network : nx.Graph
        The subsetted network FOR AN INDIVIDUAL that contains the MST
        skeleton and the extra nodes/edges up to the proportional threshold
    percent_shared_edges : float
        a calculation of edges that are in both the MST and the density base
        list. At more stringent thresholds, not all the MST edges are in the
        highest percentage of ranked edges.
    """

    n_edges_density = fam.get_edge_count(
        prop_thr)  # Same for any FCN at a given
    # thr, b/c it is based on the whole brain matrix
    thresholded_network = nx.algorithms.tree.mst.maximum_spanning_tree(G)
    mst_edges = [tuple(m)
                 for m in thresholded_network.edges()]  # For debugging
    sorted_edges, sorted_weights = sort_edge_weights(G, verbose=verbose)
    shared_edges = []
    while len(thresholded_network.edges()) < n_edges_density:
        try:
            edge = sorted_edges.pop()
            wt = sorted_weights.pop()
            if edge not in thresholded_network.edges():
                thresholded_network.add_edge(edge[0], edge[1], weight=wt)
            else:
                shared_edges.append(edge)
        except BaseException as e:
            # Rethrow the error until debugged forever.
            print(e)
            print('Error occured. Trying to add edges to a non-whole brain ',
                  'graph. If you did enter the whole brain graph, the edge ',
                  'density may need to be re-calculated based on a network ',
                  'subset.')
            print(
                f'GRAPH edges = {n_edges_density}\nMST edges = {len(mst_edges)}')
            return
        percent_shared_edges = len(shared_edges) / len(mst_edges)
    return thresholded_network, percent_shared_edges


def filter_density_based_network(thresholded_network, subgraph_network=None):
    """Input: String-based network of interest (e.g., 'frontoparietal') and
           MST + density-based thresholded network (output of add_thr_edges)
       Output: Selected network graph to be used with calculate_graph_msrs
           to determine metrics
    """
    network_parcels = get.get_network_parcels(subgraph_network)
    parcel_list = [v for v in network_parcels.values()]
    H = thresholded_network[0].subgraph(parcel_list)
    return H


def create_density_based_network(subj_idx, prop_thr):
    """Calculate the whole-brain MST for an individual and then add back high
       connectivity edges until a threshold is met for each individual"""
    # BMS Forced whole brain connectivity, so that other networks of interest
    # can be passed to funcs without overriding MST for whole brain
    mat = get.get_network_matrix('whole_brain', subj_idx)
    G = make_graph_without_nans(mat)
    thresholded_network, percent_shared_edges = add_thr_edges(
        G, prop_thr=prop_thr)
    return thresholded_network, percent_shared_edges


def calculate_graph_msrs(G, subgraph_name=None, prop_thr=None, subj=None):
    df = None  # To override option to append data
    #df = utils.get_long_format_results()
    cmplt_msrs = []
    if df is not None:
        if subgraph_name and prop_thr:
            # Eliminate the columns that have empty entries
            tmp = df.loc[(df['threshold'] == prop_thr) & (
                df['network'] == subgraph_name) &
                (df['subj_ix'] == subj), :].dropna(axis='columns')
        elif prop_thr:
            tmp = df.loc[(df['threshold'] == prop_thr) &
                         (df['subj_ix'] == subj),
                         :].dropna(axis='columns')
        else:
            tmp = df.loc[df['subj_ix'] == subj, :].dropna(axis='columns')

        if not tmp.empty:
            cmplt_msrs = tmp.columns

    # Instantiate a new graph measure dictionary
    individ_graph_msr_dict = {}
    if nx.is_connected(G):
        possible_msrs = ['nx_communities',
                         'nx_num_of_comm',
                         'modularity',
                         'shortest_path',
                         'local_efficiency']
        # Check against completed measures so that only incomplete or missing
        # analyses are run.
        to_run = [msr for msr in possible_msrs if msr not in cmplt_msrs]
        if to_run:
            communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(
                G)
            connected_dict = {'nx_communities': '[communities]',
                              'nx_num_of_comm': 'len(communities)',
                              'modularity': 'nx.algorithms.community.quality.modularity(G, communities)',
                              'shortest_path': 'nx.algorithms.shortest_paths.generic.average_shortest_path_length(G, method="dijkstra")',
                              'local_efficiency': 'nx.algorithms.efficiency_measures.local_efficiency(G)',
                              # 'mean_fc' : sum(G.degree(weight='weight'))/float(len(G))}
                              }
            for msr in to_run:
                # Should be evaluated now
                individ_graph_msr_dict[msr] = eval(connected_dict[msr])

    if not nx.is_connected(G) or subgraph_name:
        possible_msrs = ['sg_num_total_edges',
                         'sg_num_total_nodes',
                         'sg_num_connected_comp',
                         'sg_largest_component',
                         'sg_average_clustering',
                         'sg_shortest_path_length',
                         'sg_global_efficiency',
                         'mean_degree',
                         'network',
                         'threshold']
        to_run = [msr for msr in possible_msrs if msr not in cmplt_msrs]
        if to_run:
            subgraph = largest_subgraph(G)
            discnntd_dict = {
                'sg_num_total_edges': 'len(G.edges)',
                'sg_num_total_nodes': 'len(G.nodes)',
                'sg_num_connected_comp': 'nx.algorithms.components.number_connected_components(G)',
                'sg_largest_component': 'len(subgraph)',
                'sg_average_clustering': 'nx.average_clustering(subgraph)',
                'sg_shortest_path_length': 'nx.average_shortest_path_length(subgraph)',
                'sg_global_efficiency': 'nx.global_efficiency(subgraph)',
                'mean_degree': 'np.nanmean(nx.degree(G))',
                'network': 'subgraph_name',
                'threshold': 'prop_thr'}
            for msr in to_run:
                # Should be evaluated now
                individ_graph_msr_dict[msr] = eval(discnntd_dict[msr])
    return individ_graph_msr_dict


def collate_graph_measures(
        subjects=None,
        grouping_col='group',
        prop_thr=None,
        subgraph_network=None,
        multiproc=True):
    """
    Workhorse method
    """
    # Set subjects to go through the methods
    if subjects is not None:
        if isinstance(subjects, np.ndarray):
            subjects = list(subjects)
        elif isinstance(subjects, int):
            # Case from save_long_format_results when single subject is passed
            subjects = [subjects]
        elif not isinstance(subjects, list):
            # The case where group name was used for the list of subjects
            edited_dict = {
                k: v for k,
                v in shared.__dict__.items() if isinstance(
                    v,
                    list)}
            field = [k for k, v in edited_dict.items() if subjects in v]
            name_str = field[0].split('.')[-1]
            subjects = [v for k, v in shared.__dict__.items() if k ==
                        name_str][0]
    else:
        subjects = (shared.group1_indices + shared.group2_indices)
    print(f'Network: {subgraph_network}\nAnalyzing subjects: {subjects}')

    global tmp
    tmp = current_analysis(grouping_col, prop_thr, subgraph_network)
    if multiproc:
        if not subgraph_network:
            with utils.parallel_setup() as pool:
                df = pd.concat(pool.map(parallel_graph_msr, subjects))
        else:
            with utils.parallel_setup() as pool:
                df = pd.concat(
                    pool.map(parallel_subgraph_msr, subjects))
    else:
        df_list = []
        for subj in subjects:
            print(f'Working on {thr} for {subj}')
            if not subgraph_network:
                df_list.append(
                    individ_graph_msrs(
                        subj,
                        prop_thr=thr,
                        grouping_col=tmp.grouping_col))
            else:
                if isinstance(subgraph_network, str):
                    subgraph_network = [subgraph_network]
                for network in subgraph_network:
                    df_list.append(
                        individ_subgraph_msrs(
                            network,
                            subj,
                            prop_thr=thr,
                            grouping_col=tmp.grouping_col))
        df = pd.concat(df_list)
    df = df.replace({'nan', np.nan})
    return df


def parallel_graph_msr(subj):
    return individ_graph_msrs(
        subj,
        prop_thr=tmp.prop_thr,
        grouping_col=tmp.grouping_col)


def parallel_subgraph_msr(subj):
    return individ_subgraph_msrs(
        tmp.subgraph_network,
        subj,
        prop_thr=tmp.prop_thr,
        grouping_col=tmp.grouping_col)


def individ_graph_msrs(subj, prop_thr=None, grouping_col='group'):
    """ Calculate multiple graph measures for a subject at given threshold.

        Parameters
        ----------
        subj : int, index of subject
        prop_thr : float, threshold to be applied to whole brain graph
        grouping_col : string, which group subject belongs to
            (this is relevant to later analysis functions)

        Returns
        -------
        Pandas dataframe containing the subject info and graph measures

    """
    thr_G, percent_shared_edges = create_density_based_network(subj, prop_thr)
    igmd = calculate_graph_msrs(thr_G, prop_thr=prop_thr, subj=subj)
    tmp_df = pd.DataFrame(igmd, index=[subj])
    tmp_df = pd.concat([tmp_df,
                        pd.DataFrame(columns=['percent_shared_edges',
                                              'threshold',
                                              'subj_ix'])])
    tmp_df[['percent_shared_edges', 'threshold', 'subj_ix']
           ] = percent_shared_edges, prop_thr, subj
    tmp_df['network'] = 'whole_brain'
    tmp_df = utils.subject_converter(tmp_df, orig_subj_col='subj_ix')
    print(f'End {subj} {prop_thr}')
    return tmp_df


def individ_subgraph_msrs(
        subgraph_name,
        subj,
        prop_thr=None,
        grouping_col='group'):
    thr_G = create_density_based_network(subj, prop_thr)
    subgraph = filter_density_based_network(thr_G, subgraph_name)
    igmd = calculate_graph_msrs(
        subgraph,
        subgraph_name=subgraph_name,
        prop_thr=prop_thr,
        subj=subj)
    tmp_subgraph_df = pd.DataFrame(igmd, index=[subj])
    tmp_subgraph_df['subj_ix'] = subj
    tmp_subgraph_df = utils.subject_converter(
        tmp_subgraph_df, orig_subj_col='subj_ix')
    return tmp_subgraph_df


def graph_msr_group_diffs(
        network, grouping_col, prop_thr_list=np.arange(.09, 1, .1), limit_subjs=None, save=False):
    """Inputs: network is '' for whole brain, otherwise choose the name or beginning of the name for the desired network.
    grouping_col can be any categorical column, such as group, cognitive_impairment, etc.

    Output: long format data of the graph measures defined in calculate_graph_msrs for each person.
    """
    df_list = []
    for thr in tqdm(prop_thr_list):
        tmp_df = collate_graph_measures(
            network, subjects=limit_subjs, prop_thr=thr)
        df_list.append(tmp_df)
    df = pd.concat(df_list)

    if save:
        utils.save_df(df, '_long_graph_msrs.csv')
    return df


def save_long_format_results(
        output_filename,
        subjects=None,
        grouping_col='group',
        prop_thr=np.arange(.05, .99, .05).tolist(),
        networks=None,
        multiproc=True):
    """All input arguments the same as collate_graph_measures,
    plus output filepath for csv with the results for each
    subject, threshold, network, etc.
    """
    if 'long_format' not in output_filename:
        output_filename = (os.path.splitext(output_filename)[
            0] + '_long_format.csv')

    # Since calculating the graph measures now checks for previously analyzed
    # data and excludes repetitve calculations, import the
    orig_df = None
    # orig_df = utils.get_long_format_results() # Effectively turns off all
    # the addendums.
    if orig_df is None:
        orig_df = pd.DataFrame(columns=['network', 'subject', 'threshold'])

    df_list = []
    parcels = get.get_network_parcels('whole_brain')
    print(f'Testing {len(parcels)} ROIs')  # Validating
    networks = sorted(set([fcn.split("_")[0] for fcn in parcels.keys()])
                      ) + ['whole_brain'] if not networks else networks
    networks = [networks] if not isinstance(networks, list) else networks
    prop_thr = [prop_thr] if not isinstance(prop_thr, list) else prop_thr

    for network in networks:
        for thr in prop_thr:
            # Round the threshold to avoid strange, long floats.
            thr = np.round(thr, decimals=2)
            print(f'Testing {networks} at {thr}')
            # Maintain only one call to collate_graph_measures by effectively eliminating
            # subgraph network argument for whole brain.
            network = None if network in [
                'wb', 'whole_brain', 'whole brain'] else network
            df = collate_graph_measures(
                subjects=subjects,
                grouping_col=grouping_col,
                prop_thr=thr,
                subgraph_network=network,
                multiproc=multiproc)
            df_list.append(df)

            # Intentionally overwrite the file at each iteration, so that if the code crashes,
            # there is a record of the previous results.
            df_out = pd.concat(df_list).dropna(how='all', axis='columns')
            df_out = df_out.replace({'nan', np.nan})

            try:
                # Ensure data types are compatible
                for d in [orig_df, df_out]:
                    d[['subject', 'network']] = d[[
                        'subject', 'network']].astype(str)
                    d['threshold'] = d['threshold'].astype(float)
                # Get entries from original DF, but add extra information if
                # available from new analysis

                df = orig_df.merge(
                    df_out, on=[
                        'subject', 'threshold', 'network'], how='left', suffixes=(
                        '', '_x'))

                # Get the entries that are only available in the new analysis
                new_info = orig_df.merge(
                    df_out, on=[
                        'subject', 'threshold', 'network'], how='right', suffixes=(
                        '_x', ''), indicator=True)
                new_info = new_info.loc[new_info['_merge']
                                        == 'right_only', :].drop(columns='_merge')
                # Clean up any accidental duplicates
                new_info = new_info.drop(
                    new_info.filter(
                        regex='_x').columns, axis=1)
                df = df.drop(df.filter(regex='_x').columns, axis=1)

                if not new_info.empty:
                    df = pd.concat([df, new_info], copy=False)
                df = df.loc[:, ~df.columns.duplicated()]
                # By default, save_df will prepend the date of the analysis
                utils.save_df(df, output_filename)
            except KeyError:
                print('No new data to add.')


def summarize_graph_msr_group_diffs(
        df,
        grouping_col,
        limit_subjs=None,
        save=False):
    """Inputs:
        Long format dataframe created by graph_msr_group_diffs.
        grouping_col can be any categorical column, such as group, cognitive_impairment, etc.

    Output:
        stat_df = summary table of group differences derived from df. p-values are included.
    """
    df = df.replace({'nan', np.nan})
    thr_list = set(df['threshold'])
    stat_df_list = []
    msrs = [col for col in df.columns if col not in [
        'threshold', 'group', 'subj', grouping_col]]
    if grouping_col not in df.columns:
        df = utils.subject_converter(df, add_characteristics=[grouping_col])
    for thr in thr_list:
        tmp_stat_df = nss.summarize_group_differences(
            df.loc[df['threshold'] == thr], grouping_col, msrs)
        tmp_stat_df['threshold'] = thr
        stat_df_list.append(tmp_stat_df)
    stat_df = pd.concat(stat_df_list)

    if save:
        utils.save_df(stat_df, 'group_summary_graph_msrs.csv')
    return stat_df
