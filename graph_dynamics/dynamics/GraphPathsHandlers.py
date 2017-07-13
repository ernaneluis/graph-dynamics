'''
Created on Jul 4, 2017

@author: cesar
'''
import copy
import networkx as nx

def staticGraphInducedBySeries(graph_paths):
    """
    This function creates the static graph induced from the graph series
    by aggregating all the edges in one graph
    """
    static_graph = nx.Graph()
    for graph_t in graph_paths:
        static_graph.add_edges_from(graph_t.edges())
    return static_graph
    
def temporalGraphFromSeries(graph_paths):
    """
    This function creates temporal graph where
    """
    #TO DO: time in and time out should be a list, since the same edge can appear and disappear, TAKE NOTICE
    temporal_graph = nx.Graph()
    graph_0 = graph_paths[0]
    temporal_graph.add_edges_from(graph_0.edges(),time_in=0)
    for t,graph_1 in enumerate(graph_paths[1:]):
        new_edges = copy.deepcopy(graph_1)
        old_edges = copy.deepcopy(graph_0)
        
        new_edges.remove_edges_from(graph_0.edges())
        old_edges.remove_edges_from(graph_1.edges())
                
        temporal_graph.add_edges_from(new_edges.edges(),time_in=t)
        for edge_removed in old_edges.edges():
            temporal_graph[edge_removed[0]][edge_removed[1]]["time_out"] = t
        graph_0 = graph_1
    return temporal_graph

def seriesFromTemporalGraph(temporal_graph):
    return None