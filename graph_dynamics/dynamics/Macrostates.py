'''
Created on Jul 12, 2017

@author: cesar

THIS ARE THE FUNCTIONS TO BE 
CALLED BY THE DYNAMICS

ALL FUNCTION SHERE DEFINED MUST RETURN A JSON
'''
import json
import numpy as np
import networkx as nx
from graph_dynamics.embeddings import node2vec
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.networks.datatypes import VanillaGraph


def degree_distribution(Graph,*parameters):
    """
    Parameters:
    ----------
        Graph: graph object    
    """
    return Graph.get_networkx().degree()

def basic_stats(Graph,*parameters):
    """
    Parameters
    ----------
        Graph:
    
    Returns
    -------
    json_dict = {"number_of_edges":int,
                "number_of_edges":int}
    """
    
    return {"number_of_nodes":Graph.get_networkx().number_of_nodes(),
            "number_of_edges":Graph.get_networkx().number_of_edges()}

def networkx_pagerank(Graph,*parameters):
    """
    Parameters
    ----------
        Graph:
    
    Returns
    -------
    json_dict = {"number_of_edges":int,
                "number_of_edges":int}
    """ 
    return nx.pagerank(Graph.get_networkx())
    
def node2vec_macrostates(Graph,*nargs):
    """
    
    Returns
    ------
    node2vecs
    """ 
    args = nargs[0][0]
    #= Graph.get_networkx()
    nx_G = Graph.get_networkx()
    G = node2vec.Graph(nx_G, 
                       args["directed"], 
                       args["p"], 
                       args["q"])
    
    G.preprocess_transition_probs()
    
    walks = G.simulate_walks(args["num_walks"], 
                             args["walk_length"])
    
    embeddings = node2vec.learn_embeddings(walks,args)
    json_embeddings = dict(zip(embeddings.index2word,[e.tolist() for e in embeddings.syn0]))
    return json_embeddings

def evaluate_vanilla_macrostates(gd_dynamics,macrostates_names,macrostates_run_ideintifier):
    """
    """
    ALL_TIME_INDEXES, GRAPH_FILES = gd_files_handler.gd_folder_stats(gd_dynamics)
    GRAPH_FILES.sort()
    dynamics_identifier = GRAPH_FILES[0].split("_")[0]
    
    #TO DO: parallelize calls to macrostates
    for graph_filename in GRAPH_FILES:
        time_index = int(graph_filename.split("_")[-2])
        networkx_graph = nx.read_edgelist(gd_dynamics+graph_filename)
        Vanilla =  VanillaGraph(dynamics_identifier,{"None":None},networkx_graph)
        macrostate_filename = gd_dynamics+"{0}_mGD_{1}_{2}_.gd".format(dynamics_identifier,
                                                               macrostates_run_ideintifier,                    
                                                               time_index)
        macrostate_json = {}
        for macrostate_function in macrostates_names:
            macrostate_function_name = macrostate_function[0]
            macrostate_function_parameters = macrostate_function[1]
            macrostate_json[macrostate_function_name] = macrostate_function_dictionary[macrostate_function_name](Vanilla,*macrostate_function_parameters)
                                   
        with open(macrostate_filename,"w") as outfile:
            json.dump(macrostate_json, outfile)
            
#=====================================
#
#=====================================

macrostate_function_dictionary = {"degree_distribution":degree_distribution,
                                  "node2vec_macrostates":node2vec_macrostates,
                                  "basic_stats":basic_stats,
                                  "pagerank":networkx_pagerank}