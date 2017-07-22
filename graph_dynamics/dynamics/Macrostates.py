'''
Created on Jul 12, 2017

@author: cesar

THIS ARE THE FUNCTIONS TO BE
CALLED BY THE DYNAMICS EVOLVE FUNCTION

ALL FUNCTION SHERE DEFINED MUST RETURN A JSON
'''
import sys
import json
import numpy as np
import networkx as nx
from graph_dynamics.embeddings import node2vec
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.networks.datatypes import VanillaGraph
from graph_dynamics.communities.bigclam import BigClam

from graph_dynamics.utils.timeseries_utils import createWindows


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
    json: node2vecs
    """
    args = nargs[0]
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


def new_nodes(GRAPH_LIST,*param):
    """
    Parameters
    ----------
    GRAPH_LIST: list of Graoh objects

    Return
    ------
    """
    nodes_1 = set(GRAPH_LIST[1].get_networkx().nodes())
    nodes_0 = set(GRAPH_LIST[0].get_networkx().nodes())
    newNodes = nodes_1.difference(nodes_0)
    number_of_new = len(newNodes)
    return {"new_nodes":list(newNodes),"number_of_new_nodes":number_of_new}


def evaluate_vanilla_macrostates(gd_directory,macrostates_names,macrostates_run_ideintifier):
    """
    This function evaluates macrostates in gd directories with no states

    Parameters
    ----------
    gd_directory: gd directory name of dynamics
    macrostates_names: list
                    [(macro-string1,macro_parameters1),...,(macro-string_M,macro_parameters_M)]
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory)
    #check if dynamics parameters are complete
    dynamics_identifier = DYNAMICS_PARAMETERS["dynamics_identifier"]
    #TO DO: parallelize calls to macrostates
    for time_index in ALL_TIME_INDEXES:
        graph_filename = "{0}_gGD_{1}_.gd".format(dynamics_identifier,time_index)
        try:
            print "Evaluating Time {0} for {1}".format(time_index,macrostates_run_ideintifier)
            #print graph_filename
            networkx_graph = nx.read_edgelist(gd_directory+graph_filename)

            Vanilla =  VanillaGraph(dynamics_identifier,{"None":None},networkx_graph)
            macrostate_filename = gd_directory+"{0}_mGD_{1}_{2}_.gd".format(dynamics_identifier,
                                                                   macrostates_run_ideintifier,
                                                                   time_index)
            macrostate_json = {}
            for macrostate_function in macrostates_names:
                macrostate_function_name = macrostate_function[0]
                macrostate_function_parameters = macrostate_function[1]
                macrostate_json[macrostate_function_name] = macrostate_function_dictionary[macrostate_function_name](Vanilla,*macrostate_function_parameters)

            #print macrostate_filename
            with open(macrostate_filename,"w") as outfile:
                json.dump(macrostate_json, outfile)
        except:
            print sys.exc_info()
            print "Problem with time index {0}".format(time_index)
            print "Graph ",graph_filename

def get_vanilla_graph(gd_directory,dynamics_identifier,time_index):
    """
    """
    graph_filename = "{0}_gGD_{1}_.gd".format(dynamics_identifier,time_index)
    networkx_graph = nx.read_edgelist(gd_directory+graph_filename)
    Vanilla =  VanillaGraph(dynamics_identifier,{"None":None},networkx_graph)
    return Vanilla

def ouput_macrostate_json(gd_directory,dynamics_identifier,macrostates_names,macrostates_run_ideintifier,time_index,GRAPHS_IN_WINDOW):
    """
    """
    macrostate_filename = gd_directory+"{0}_mGD_{1}_{2}_.gd".format(dynamics_identifier,
                                                                    macrostates_run_ideintifier,
                                                                    time_index)
    macrostate_json = {}
    for macrostate_function in macrostates_names:
        macrostate_function_name = macrostate_function[0]
        macrostate_function_parameters = macrostate_function[1]
        macrostate_json[macrostate_function_name] = macrostate_function_dictionary[macrostate_function_name](GRAPHS_IN_WINDOW,*macrostate_function_parameters)

    with open(macrostate_filename,"w") as outfile:
        json.dump(macrostate_json, outfile)

def evaluate_vanilla_macrostates_window(gd_directory,macrostates_names,macrostates_run_ideintifier,window=2,rolling=False):
    """
    This function evaluates macrostates in gd directories with no states

    Parameters
    ----------
    gd_directory: gd directory name of dynamics
    macrostates_names: list
                    [(macro-string1,macro_parameters1),...,(macro-string_M,macro_parameters_M)]
    macrostates_run_identifier: string
        identifier fot the file for all the macrostates given in the list
    window: int
        is the number of graphs requiered for the calculation of the macrostate
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory)

    #check if dynamics parameters are complete
    WINDOWS = createWindows(ALL_TIME_INDEXES, window, rolling)
    dynamics_identifier = DYNAMICS_PARAMETERS["dynamics_identifier"]
    if rolling:
        GRAPHS_IN_WINDOW = []
        for time_index in WINDOWS[0]:
            GRAPHS_IN_WINDOW.append(get_vanilla_graph(gd_directory,dynamics_identifier,time_index))
        ouput_macrostate_json(gd_directory,dynamics_identifier,macrostates_names,macrostates_run_ideintifier,time_index,GRAPHS_IN_WINDOW)

    #TO DO: parallelize calls to macrostates
    for window in WINDOWS[1:]:
        print "Evaluating Time {0} for {1}".format(time_index,macrostates_run_ideintifier)
        try:
            if rolling:
                GRAPHS_IN_WINDOW.pop(0)
                time_index = window[-1]
                GRAPHS_IN_WINDOW.append(get_vanilla_graph(gd_directory,dynamics_identifier,time_index))
            else:
                GRAPHS_IN_WINDOW = []
                for time_index in window:
                    GRAPHS_IN_WINDOW.append(get_vanilla_graph(gd_directory,dynamics_identifier,time_index))
            ouput_macrostate_json(gd_directory,dynamics_identifier,macrostates_names,macrostates_run_ideintifier,time_index,GRAPHS_IN_WINDOW)
        except:
            print "Problem with time index {0}".format(time_index)

def bigclam(Graph,*nargs):
    """
        Parameters
        ----------
            Graph:

        Returns
        -------
        json_dict = {
                    "weights_matrix": matrix: [node_index][community_index],  //Weight for affiliation
                    "community_cluster": list: [community_index_x, community_index_y, ..., community_index_z] // return which nodes belongs to   cluster index x
                    "colors":  list: [color_of_community_index_1, color_of_community_index_2, ..., color_of_community_index_n]
                    }
    """

    args = nargs[0]
    bigClamObj = BigClam(Graph, maxNumberOfIterations=args["max_number_of_iterations"], error=args["error"], beta=args["beta"])
    out = {
            "weights_matrix":       json.dumps(bigClamObj.F.tolist()),
            "community_cluster":    list(bigClamObj.community_cluster),
            "colors":               list(bigClamObj.values),
           }
    return out
#========================================================================================================================
# THE FOLLOWING DICTIONARY HOLDS ALL MACROSTATES WHICH CAN BE CALLED BY THE EVOLUTION FUNCTION OF GRAPH DYNAMICS
#========================================================================================================================

macrostate_function_dictionary = {
                                  "degree_distribution":degree_distribution,
                                  "node2vec_macrostates":node2vec_macrostates,
                                  "basic_stats":basic_stats,
                                  "pagerank":networkx_pagerank,
                                  "new_nodes":new_nodes,
                                  "bigclam":bigclam
                                  }
