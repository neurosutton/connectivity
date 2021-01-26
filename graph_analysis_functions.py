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

# TODO refactor homegrown calls to jive with new GH-based code (i.e., "tan", "faf" with appropriate functions)
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
    else:
        print(kwargs.keys())

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
    threshold = [threshold] if type(threshold) == float else threshold
    bckgrd_data = get.subj_data()
    for value in threshold:
        for idx in range(len(bckgrd_data)):
            print("\r>> Calculating for subject {} at threshold={}".format(idx, value), end='')
            df_pos = len(graph_df) # looping index. Better way?
            subj_data = np.expand_dims(conn_data[:, :, idx].copy(), axis=2)
            mask = get.get_prop_thr_edges(value, conn_data=subj_data)[:, :, 0]
            matrix = get.get_network_matrix(network_name=network, subj_idx=0,
                                            network_mask=mask,
                                            conn_data=subj_data)
            graph = make_graph_without_nans(matrix)
            subgraph = largest_subgraph(graph)
            # TODO Translate to pandas style with text-based labels, rather than indexed numbers that are not easily read with multiple thresholds for the same subjects.
            graph_df.at[df_pos, 'subject'] = bckgrd_data['subject'][idx]
            graph_df.at[df_pos, 'group'] = bckgrd_data['group'][idx]
            graph_df.at[df_pos, 'threshold'] = value
            graph_df.at[df_pos, 'global_mean'] = np.nanmean(matrix)
            graph_df.at[df_pos, 'density'] = nx.density(graph)
            graph_df.at[df_pos, 'largest_component'] = len(subgraph)
            graph_df.at[df_pos, 'average_clustering'] = nx.average_clustering(subgraph)
            graph_df.at[df_pos, 'shortest_path_length'] = nx.average_shortest_path_length(subgraph)
            graph_df.at[df_pos, 'global_efficiency'] = nx.global_efficiency(subgraph)
            graph_df.at[df_pos, 'mean_degree'] = np.nanmean(nx.degree(graph))
            # graph_df.at[df_pos, 'local_efficiency'] = nx.local_efficiency(subgraph)
            # graph_df.at[df_pos, 'average_node_connectivity'] = nx.average_node_connectivity(graph)
            graph_df.at[df_pos, 'global_edges'] = len(graph.edges())
            graph_df.at[df_pos, 'Q'] = nx_comm.modularity(subgraph, nx_comm.label_propagation_communities(subgraph))
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
def get_pos_dict(network):
    bnv = bnv_prep.bnv_analysis(network=network, prop_thr=None)
    network_locs = bnv.limit_labels(network=network)
    roi_dict = OrderedDict()
    for n,net in enumerate(network_locs['SuttonLabel']):
        roi_dict[n] = network_locs.loc[network_locs['SuttonLabel']==net,['x','y']].values[0]
    pos ={}
    for k,v in roi_dict.items():
        pos[k]=v

    return pos

def sort_edge_weights(G):
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


def create_density_based_network(network, subj_idx, prop_thr):
    """Calculate the whole-brain MST for an individual and then add back high connectivity edges until a threshold is met for each individual"""
    #common_mat = get.get_cohort_network_matrices(network, all_the_indices)
    mat = get.get_network_matrix(network,subj_idx)
    G = make_graph_without_nans(mat)
    thresholded_network,percent_shared_edges = add_thr_edges(G,prop_thr=prop_thr)
    return thresholded_network,percent_shared_edges

def calculate_graph_msrs(G):
    individ_graph_msr_dict = {}
    individ_graph_msr_dict['avg_path_len'] = nx.algorithms.shortest_paths.generic.average_shortest_path_length(G)
    #individ_graph_msr_dict['global_efficiency'] = nx.algorithms.efficiency_measures.global_efficiency(G) # Not used currently for computational efficiency, given that avg path length is the inverse
    individ_graph_msr_dict['clustering'] = nx.algorithms.average_clustering(G)
    #individ_graph_msr_dict['smallWorld'] = nx.algorithms.omega(G) #computationally intensive
    #individ_graph_msr_dict['richClub'] = nx.algorithms.rich_club_coefficient(G) # returns dictionary with hubs and RC value
    neigh_deg = nx.algorithms.average_neighbor_degree(G)
    individ_graph_msr_dict['avg_neigh_deg'] = np.mean(np.array([n for n in neigh_deg.values()]))

    return individ_graph_msr_dict

def collate_graph_measures(network, subjects=None, prop_thr = None):
    df_list = []
    if subjects: 
       if not isinstance(subjects,list):
           field = [k for k,v in shared.__dict__.items() if v == subjects]
           name_str = field[0].split('.')[-1]+'_indices'
           subjects = [v for k,v in shared.__dict__.items() if k == name_str][0]
           print(subjects)
    else:
       subjects =(shared.group1_indices+shared.group2_indices)
    for subj in tqdm(subjects):
        thr_G, percent_shared_edges = create_density_based_network(network, subj, prop_thr)
        igmd = calculate_graph_msrs(thr_G)
        tmp_df = pd.DataFrame(igmd, index=[subj])
        tmp_df[['percent_shared_edges','threshold','group']] = percent_shared_edges,prop_thr, utils.match_subj_group(subj)
        df_list.append(tmp_df)

    df = pd.concat(df_list).reset_index()
    df.rename(columns={'index':'subj'}, inplace=True)
    return df

def graph_msr_group_diffs(network, grouping_col, prop_thr_list=np.arange(.85,1,.01), limit_subjs=None):
    """Inputs: network is '' for whole brain, otherwise choose the name or beginning of the name for the desired network.
    grouping_col can be any categorical column, such as group, cognitive_impairment, etc.

    Output: df = long format data of the graph measures defined in calculate_graph_msrs for each person.
    stat_df = summary table of group differences derived from df. p-values are included.
    """
    df_list = []
    stat_df_list = []
    for thr in prop_thr_list:
        tmp_df = collate_graph_measures(network,subjects=limit_subjs, prop_thr = thr)
        msrs = [col for col in tmp_df.columns if col not in ['threshold','group','subj']]
        tmp_stat_df = nss.summarize_group_differences(tmp_df, grouping_col, msrs)
        tmp_stat_df['threshold'] = thr
        df_list.append(tmp_df)
        stat_df_list.append(tmp_stat_df)
    df = pd.concat(df_list)
    stat_df = pd.concat(stat_df_list)
    return df, stat_df
