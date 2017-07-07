'''
Created on Jul 3, 2017

@author: Can
'''
import networkx as nx
import snap

def snap_to_nx(GS):
    GN = nx.DiGraph();
    for EI in GS.Edges():
        GN.add_edge(EI.GetSrcNId(), EI.GetDstNId())
    return GN

def nx_to_snap(GN):
    GS = snap.TNGraph.New();
    for ni in GN.nodes():
        GS.AddNode(ni)
    for f,t in GN.edges():
            GS.AddEdge(f,t)
    return GS