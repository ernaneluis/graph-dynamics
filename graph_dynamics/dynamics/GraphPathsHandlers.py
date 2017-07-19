'''
Created on Jul 4, 2017

@author: cesar
'''
import copy
import numpy as np
import networkx as nx
import datetime
#from samba.dcerpc.atsvc import DAYSOFWEEK_WEDNESDAY

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

def seriesFromTemporalGraph(gd_folder,dynamics_identifier,temporalFileName,stepsInGraph=7,parseunix=False):
    """
    """
    temporal_edges = np.loadtxt(temporalFileName,delimiter=" ")
    if parseunix:
        days = map(datetime.datetime.fromtimestamp,temporal_edges[:,2])
        minday = min(days)
        maxday = max(days)
        print maxday
        print minday
        print "Total Day Difference: ",(maxday - minday).days
        days = [(a-minday).days for a in days]
        raise Exception
    else:
        days = temporal_edges[:,2]
    minDay = int(min(days))
    maxDay = int(max(days))
    for time_index, current_day in enumerate(range(minDay,maxDay,stepsInGraph)):
        graph_file_name  = gd_folder+"{0}_gGD_{1}_.gd".format(dynamics_identifier,time_index)
        current_edges = np.take(temporal_edges,np.where(days < current_day)[0],axis=0)[:,[0,1]]
        np.savetxt(graph_file_name,current_edges)
