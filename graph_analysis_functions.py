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
import shared
import fmri_analysis_get_data as get

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
