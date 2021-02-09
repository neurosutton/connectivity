import networkx as nx
import numpy as np
from networkx.algorithms import community
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from importlib import reload
import seaborn as sns
from scipy.stats import ttest_ind
import networkx.algorithms.community as nx_comm
from sklearn import metrics
from random import shuffle
from networkx.algorithms import community

# BMS
from collections import OrderedDict
import shared
import fmri_analysis_get_data as get
import fmri_analysis_manipulations as fam
import fmri_analysis_bnv_prep as bnv_prep
import fmri_analysis_utilities as utils
utils.check_data_loaded()
import nia_stats_and_summaries as nss
from tqdm import tqdm
#>>>END BMS

# CREATE A FUNCTION TO SIMPLIFY PLOTTING and GRAPH MEASURES
def plot_weighted_graph(gw, **kwargs):
    eweights = [d['weight'] for (u, v, d) in gw.edges(data=True)]
    options = {
        "edge_color": eweights,
        "width": 1,
        "node_size": 200,
        "node_color": 'yellow',
        "edge_cmap": plt.cm.rainbow,
        "edge_vmin": 0,
        "edge_vmax": 1,
        "with_labels" :  False
    }
    if 'pos' in kwargs.keys():
        options['pos'] = kwargs['pos']

    if 'node_weights' in kwargs.keys():
        add_node_weights(gw, kwargs['node_weights'][0], kwargs['node_weights'][1])
        options['node_color'] = [v for v in nx.get_node_attributes(gw,kwargs['node_weights'][0])]

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw(gw, ax=ax, **options)
    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax)
    plt.show()

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


def get_graph_measures(network='', threshold=(1-0.008)):
    """
    Summarizes individuals' graphs and concatenates the summary measures into a dataframe for further analysis.
    """
    conn_data = get.get_conn_data()
    # conn_data[:, :, 0].shape
    graph_df = pd.DataFrame(columns=['subject', 'group', 'density', 'largest_component',
                                     'global_mean', 'threshold', 'average_clustering',
                                     'shortest_path_length', 'global_edges', 'Q',
                                     'global_efficiency', 'local_efficiency',
                                     'average_node_connectivity', 'mean_degree'])
    graph_df = utils.subject_converter(graph_df,orig_subj_col='subject') # BMS subject index to subject label
    threshold = [threshold] if type(threshold) == float else threshold
    bckgrd_data = get.subj_data()
    bckgrd_data = utils.subject_converter(bckgrd_data,orig_subj_col='subject')
    for value in threshold:
        for subj in set(bckgrd_data['subject']):
            idx = bckgrd_data.loc[((bckgrd_data['subject']==subj) and (bckrd_data['threshold']==value)),'subject'].index()
            print("\r>> Calculating for subject {} at threshold={}".format(idx, value), end='')
            subj_data = np.expand_dims(conn_data[:, :, idx].copy(), axis=2)
            mask = get.get_prop_thr_edges(value, conn_data=subj_data)[:, :, 0]
            matrix = get.get_network_matrix(network_name=network, subj_idx=0,
                                            network_mask=mask,
                                            conn_data=subj_data)
            graph = make_graph_without_nans(matrix)
            subgraph = largest_subgraph(graph)
            
            # TODO Translate to pandas style with text-based labels, rather than indexed numbers that are not easily read with multiple thresholds for the same subjects.
            graph_df.loc[idx, 'subject'] = subject
            graph_df.loc[idx, 'group'] = bckgrd_data['group'][idx]
            graph_df.loc[idx, 'threshold'] = value
            graph_df.loc[idx, 'global_mean'] = np.nanmean(matrix)
            graph_df.loc[idx, 'density'] = nx.density(graph)
            graph_df.loc[idx, 'largest_component'] = len(subgraph)
            graph_df.loc[idx, 'average_clustering'] = nx.average_clustering(subgraph)
            graph_df.loc[idx, 'shortest_path_length'] = nx.average_shortest_path_length(subgraph)
            graph_df.loc[idx, 'global_efficiency'] = nx.global_efficiency(subgraph)
            graph_df.loc[idx, 'mean_degree'] = np.nanmean(nx.degree(graph))
            # graph_df.at[df_pos, 'local_efficiency'] = nx.local_efficiency(subgraph)
            # graph_df.at[df_pos, 'average_node_connectivity'] = nx.average_node_connectivity(graph)
            graph_df.loc[idx, 'global_edges'] = len(graph.edges())
            graph_df.loc[idx, 'Q'] = nx_comm.modularity(subgraph, nx_comm.label_propagation_communities(subgraph))
    return graph_df


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
        auc_df.at[idx, 'group'] = df[new_df['subject']==subject]['group'].tolist()[0]
        # print(new_df[new_df['subject']==subject]['group'][0])
    return auc_df


def mean_auc_by_group(df):
    results = {}
    for group in set(df['group']):
        results[group] = np.mean(df[df['group']==group]['auc'])
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
    new_df['average_clustering'] = pd.to_numeric(new_df['average_clustering'], downcast='float')
    new_df['shortest_path_length'] = pd.to_numeric(new_df['shortest_path_length'], downcast='float')
    new_df['global_efficiency'] = pd.to_numeric(new_df['global_efficiency'], downcast='float')
    new_df['mean_degree'] = pd.to_numeric(new_df['mean_degree'], downcast='float')
    new_df['Q'] = pd.to_numeric(new_df['Q'], downcast='float')
    return new_df


def compare_measures(graph_df, measure):
    return ttest_ind(graph_df[graph_df['group']=='hc'][measure],
              graph_df[graph_df['group']=='eses'][measure])


#>>>BMS
class current_analysis():
    """Local class to pass some preset fields to the parallelization functions."""
    def __init__(self, network='',grouping_col='group',prop_thr=None, subgraph_network=None):
        self.network = network
        self.grouping_col = grouping_col
        self.prop_thr = prop_thr
        self.subgraph_network = subgraph_network

def get_position_dict(network):
    """Add position information to each node using the coordinates in a dataframe with ROI labels and coordinates.
    Use with plot_weighted_graphs to represent the results in an axial super glass brain.
    Currently, not flexible as the specific label column is hard coded."""
    bnv = bnv_prep.bnv_analysis(network=network, prop_thr=None)
    network_locs = bnv.limit_labels(network=network)
    roi_dict = OrderedDict()
    for n,net in enumerate(network_locs['SuttonLabel']):
        roi_dict[n] = network_locs.loc[network_locs['SuttonLabel']==net,['x','y']].values[0]
    position_dict ={}
    for k,v in roi_dict.items():
        position_dict[k]=v

    return position_dict

def add_node_weights(G, msr_name, nx_func):
    """Inputs: G = graph
    msr_name = key entry for nodal weighting
    nx_func = NetworkX function to generate the weights per node
    """
    msr_dict = nx_func
    for n in G.nodes:
        G.nodes[n][msr_name] = msr_dict[n]

def sort_edge_weights(G):
    """Helper function. Extract the weights, sort them, and find the matching values for a sorted list. Needed for percentile thresholding to supplement the MST selection.
    """
    weights_dict = {}
    for u,v,weight in G.edges.data("weight"):
        weights_dict[weight] = [u,v]

    sorted_weights = sorted(list(weights_dict.keys()))
    if sorted_weights:
        sorted_edges = sorted([])
        for wt in sorted_weights:
            sorted_edges.append(weights_dict[wt])
        return sorted_edges, sorted_weights
    else:
        raise ValueError('Graph did not have sortable weights.\n')


def add_thr_edges(G, prop_thr=None):
    """
    Inputs: the graph, a proportional threshold (optional) for adding nodes to the MST result.
    Outputs: The subsetted network FOR AN INDIVIDUAL that contains the MST skeleton and the extra nodes/edges up to the proportional threshold; a calculation of edges that are in both the MST and the density base list. At more stringent thresholds, not all the MST edges are in the highest percentage of ranked edges.
    """
    n_edges_density = fam.get_edge_count(prop_thr)
    thresholded_network = nx.algorithms.tree.mst.maximum_spanning_tree(G)
    mst_edges = [tuple(m) for m in thresholded_network.edges()]
    sorted_edges, sorted_weights = sort_edge_weights(G)
    shared_edges=[]
    while len(thresholded_network.edges()) < n_edges_density:
        try:
            edge = sorted_edges.pop()
            wt = sorted_weights.pop()
            if edge not in thresholded_network.edges():
                thresholded_network.add_edge(edge[0],edge[1],weight=wt)
            else:
                shared_edges.append(edge)
        except:
            print('Likely the Graph needs to be for whole brain or the edge density needs to be re-calculated based on a network subset.')
        percent_shared_edges = len(shared_edges)/len(mst_edges)
    return thresholded_network,percent_shared_edges

def filter_density_based_network(thresholded_network, subgraph_network=None):
    """Input: String-based network of interest (e.g., 'frontoparietal') and MST + density-based thresholded network (output of add_thr_edges)
    Output: Selected network graph to be used with calculate_graph_msrs to determine metrics
    """
    network_parcels = get.get_network_parcels(subgraph_network)
    parcel_list = [v for v in network_parcels.values()]
    H = thresholded_network.subgraph(parcel_list)
    return H

def create_density_based_network(network, subj_idx, prop_thr):
    """Calculate the whole-brain MST for an individual and then add back high connectivity edges until a threshold is met for each individual"""
    #common_mat = get.get_cohort_network_matrices(network, all_the_indices)
    mat = get.get_network_matrix(network,subj_idx)
    G = make_graph_without_nans(mat)
    thresholded_network,percent_shared_edges = add_thr_edges(G,prop_thr=prop_thr)
    return thresholded_network,percent_shared_edges


def calculate_graph_msrs(G):
    individ_graph_msr_dict = {}
    if nx.is_connected(G):
        individ_graph_msr_dict['gm_shortest_path'] = nx.algorithms.shortest_paths.generic.average_shortest_path_length(G, method='dijkstra')
        individ_graph_msr_dict['gm_local_efficiency'] = nx.algorithms.efficiency_measures.local_efficiency(G)
    else:
        individ_graph_msr_dict['num_total_edges'] = len(G.edges)
        individ_graph_msr_dict['num_total_nodes'] = len(G.nodes)
        individ_graph_msr_dict['num_connected_comp'] = nx.algorithms.components.number_connected_components(G)

    return individ_graph_msr_dict


def collate_graph_measures(network, subjects=None, grouping_col='group',prop_thr=None, subgraph_network=None):
    if subjects is not None:
       if not isinstance(subjects,list):
           field = [k for k,v in shared.__dict__.items() if v == subjects]
           name_str = field[0].split('.')[-1]+'_indices'
           subjects = [v for k,v in shared.__dict__.items() if k == name_str][0]
    else:
       subjects =(shared.group1_indices+shared.group2_indices)
    global tmp
    tmp = current_analysis(network, grouping_col, prop_thr, subgraph_network)
    with utils.parallel_setup() as pool:
        df = pd.concat(pool.map(parallel_graph_msr,subjects))

    if subgraph_network:
        with utils.parallel_setup() as pool:
            df_subgraph = pd.concat(pool.map(parallel_subgraph_msr, subjects))
        df = df.merge(df_subgraph, on = 'subj_ix')
    return df

def parallel_graph_msr(subj):
    return individ_graph_msrs(tmp.network, subj, prop_thr=tmp.prop_thr, grouping_col=tmp.grouping_col)

def parallel_subgraph_msr(subj):
    return individ_subgraph_msrs(tmp.network, tmp.subgraph_network, subj, prop_thr=tmp.prop_thr, grouping_col=tmp.grouping_col)


def individ_graph_msrs(network, subj, prop_thr=None, grouping_col='group'):
    thr_G, percent_shared_edges = create_density_based_network(network, subj, prop_thr)
    igmd = calculate_graph_msrs(thr_G)
    tmp_df = pd.DataFrame(igmd, index=[subj])
    tmp_df[['percent_shared_edges','threshold','subj_ix','network']] = percent_shared_edges,prop_thr,subj, network
    tmp_df = utils.subject_converter(tmp_df,orig_subj_col='subj_ix')
    print(f'End {subj} {prop_thr}')
    return tmp_df

def individ_subgraph_msrs(network, subgraph, subj, prop_thr=None, grouping_col='group'):
    thr_G, percent_shared_edges = create_density_based_network(network, subj, prop_thr)
    subgraph = filter_density_based_network(thr_G, subgraph)
    igmd = calculate_graph_msrs(subgraph)
    tmp_subgraph_df = pd.DataFrame(igmd, index=[subj])
    tmp_subgraph_df['subj_ix'] = subj
    tmp_subgraph_df = utils.subject_converter(tmp_subgraph_df,orig_subj_col='subj_ix')
    return tmp_subgraph_df


def graph_msr_group_diffs(network, grouping_col, prop_thr_list=np.arange(.09,1,.1), limit_subjs=None, save=False):
    """Inputs: network is '' for whole brain, otherwise choose the name or beginning of the name for the desired network.
    grouping_col can be any categorical column, such as group, cognitive_impairment, etc.

    Output: long format data of the graph measures defined in calculate_graph_msrs for each person.
    """
    df_list = []
    for thr in tqdm(prop_thr_list):
        tmp_df = collate_graph_measures(network,subjects=limit_subjs, prop_thr = thr)
        df_list.append(tmp_df)
    df = pd.concat(df_list)

    if save:
        utils.save_df(df, 'long_graph_msrs.csv')
    return df


def summarize_graph_msr_group_diffs(df, grouping_col, limit_subjs=None, save=False):
    """Inputs:
        Long format dataframe created by graph_msr_group_diffs.
        grouping_col can be any categorical column, such as group, cognitive_impairment, etc.

    Output:
        stat_df = summary table of group differences derived from df. p-values are included.
    """
    thr_list = set(df['threshold'])
    stat_df_list = []
    msrs = [col for col in df.columns if col not in ['threshold','group','subj',grouping_col]]
    if grouping_col not in df.columns:
        df = utils.subject_converter(df,add_characteristics=[grouping_col])
    for thr in thr_list:
        tmp_stat_df = nss.summarize_group_differences(df.loc[df['threshold']==thr], grouping_col, msrs)
        tmp_stat_df['threshold'] = thr
        stat_df_list.append(tmp_stat_df)
    stat_df = pd.concat(stat_df_list)

    if save:
        utils.save_df(stat_df,'group_summary_graph_msrs.csv')
    return stat_df
