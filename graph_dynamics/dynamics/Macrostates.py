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

import multiprocessing
from graph_dynamics.embeddings import node2vec
from graph_dynamics.embeddings import deep_walk
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.networks.datatypes import VanillaGraph
from graph_dynamics.communities.bigclam import BigClam

from graph_dynamics.utils.timeseries_utils import createWindows


def degree_nodes(Graph,*parameters):
    """
    Parameters:
    ----------
        Graph: graph object

     The node degree is the number of edges adjacent to that node.
    Returns
    -------
    nd : dictionary, A dictionary with nodes as keys and degree as values

    """
    print "Computing degree_nodes Macro..."
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
    print "Computing basic_stats Macro..."
    return {"number_of_nodes":Graph.get_networkx().number_of_nodes(),
            "number_of_edges":Graph.get_networkx().number_of_edges()}


def advanced_stats(Graph,*parameters):
    """
    Parameters
    ----------
        Graph:

    Returns
    -------
    json_dict = {
                "degree_of_distribution":dict:
                "clustering_coefficient":dict: Compute the clustering coefficient for nodes.
                "triangles":             dict: Number of triangles keyed by node label.
                "degree_centrality":     dict: Compute the degree centrality for nodes.
                }

    https://networkx.readthedocs.io/en/stable/tutorial/tutorial.html#graph-attributes
    https://networkx.readthedocs.io/en/stable/reference/generated/networkx.algorithms.cluster.clustering.html#networkx.algorithms.cluster.clustering
    https://networkx.readthedocs.io/en/stable/reference/generated/networkx.algorithms.cluster.triangles.html#networkx.algorithms.cluster.triangles
    https://networkx.readthedocs.io/en/stable/reference/generated/networkx.algorithms.centrality.degree_centrality.html#networkx.algorithms.centrality.degree_centrality




     "clustering_coefficient":nx.clustering(Graph.get_networkx()),
    """

    print "Computing advanced_stats Macro..."

    return {
            "max_degree_nodes":      max( nx.degree(Graph.get_networkx()).values() ),
            "total_triangles":             sum(sorted(nx.triangles(Graph.get_networkx()).values(), reverse=True)),
            }

def degree_centrality(Graph,*parameters):
    """
    Parameters
    ----------
        Graph:

    Returns
    -------
     "degree_centrality":     dict: Compute the degree centrality for nodes.

    https://networkx.readthedocs.io/en/stable/reference/generated/networkx.algorithms.centrality.degree_centrality.html#networkx.algorithms.centrality.degree_centrality

    """

    # return dict(zip(range(Graph.get_number_of_nodes()), bigClamObj.F.tolist()))
    print "Computing degree_centrality Macro..."
    return nx.degree_centrality(Graph.get_networkx())

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

def node2vec_online_macrostates(GRAPH_LIST,*nargs):
    """
    Parameters
    ----------
    GRAPH_LIST: list of Graoh objects

    Return
    ------
    """
    args = nargs[0]
    model = args['model']

    nex_G = GRAPH_LIST[1].get_networkx()
    pre_G = GRAPH_LIST[0].get_networkx()

    new_nodes = set(nex_G.nodes()).difference(pre_G.nodes())
    new_nodes = list(new_nodes)
    
    G = node2vec.Graph(nex_G,
                       args["directed"],
                       args["p"],
                       args["q"])

    G.preprocess_transition_probs()
    walks = G.simulate_walks(args["num_walks"],
                             args["walk_length"],
                             nodes=new_nodes)

    model.build_vocab([map(str, new_nodes)], update=True)
    model.train(walks, total_examples=model.corpus_count, epochs=model.iter)

    json_embeddings = dict(zip(model.wv.index2word,[e.tolist() for e in model.wv.syn0]))
    return json_embeddings

def new_nodes(GRAPH_LIST,*param):
    """
    Parameters
    ----------
    GRAPH_LIST: list of Graoh objects

    Return
    ------
    """
    print "Computing new_nodes Macro..."

    nodes_1 = set(GRAPH_LIST[1].get_networkx().nodes())
    nodes_0 = set(GRAPH_LIST[0].get_networkx().nodes())
    newNodes = nodes_1.difference(nodes_0)
    number_of_new = len(newNodes)
    return {"new_nodes":list(newNodes),"number_of_new_nodes":number_of_new}



def deepwalk_online(GRAPH_LIST, *nargs):
    """
    Parameters
    ----------
    GRAPH_LIST: list of Graoh objects

    Return
    ------
    """
    args = nargs[0]
    model = args['model']


    nex_G = GRAPH_LIST[1].get_networkx()
    pre_G = GRAPH_LIST[0].get_networkx()

    new_nodes = set(nex_G.nodes()).difference(pre_G.nodes())

    walks = deep_walk.walks(nex_G, number_of_walks=args['number_of_walks'], walk_length=args['walk_length'], start_nodes=new_nodes)
    walks = [map(str, walk) for walk in walks]

    # is_update = True
    # if not model.wv.vocab:
    #     is_update = False

    model.build_vocab([map(str, new_nodes)], update=True)
    model.train(walks, total_examples=model.corpus_count, epochs=model.iter)

    json_embeddings = dict(zip(model.wv.index2word,[e.tolist() for e in model.wv.syn0]))
    return json_embeddings


def evaluate_vanilla_macrostates_parallel(gd_directory,macrostates_names,macrostates_run_ideintifier,number_of_workers=3):
    """
    This function evaluates macrostates in gd directories with no states

    Parameters
    ----------
    gd_directory: gd directory name of dynamics
    macrostates_names: list
                    [(macro-string1,macro_parameters1),...,(macro-string_M,macro_parameters_M)]
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory)
    ALL_TIME_INDEXES.sort()
    #TODO: check if dynamics parameters are complete
    dynamics_identifier = DYNAMICS_PARAMETERS["dynamics_identifier"]
    #TODO: parallelize calls to macrostates

    N = len(ALL_TIME_INDEXES)
    if (N % number_of_workers) != 0:
        steps = np.concatenate([np.repeat(number_of_workers,N / number_of_workers),np.array([N % number_of_workers])])
    else:
        steps = np.repeat(number_of_workers,N / number_of_workers)


    current_index = 0
    for step in steps:
        #============================================
        #COLLECT GRAPHS FOR PARALLEL WORKERS
        #============================================
        VANILLA_GRAPHS = []
        for time_index in range(current_index,current_index+step):
            graph_filename = "{0}_gGD_{1}_.gd".format(dynamics_identifier,time_index)
            try:
                print "Evaluating Time {0} for {1}".format(time_index,macrostates_run_ideintifier)
                #print graph_filename
                networkx_graph = nx.read_edgelist(gd_directory+graph_filename)
                Vanilla =  VanillaGraph(dynamics_identifier,{"None":None},networkx_graph)
                VANILLA_GRAPHS.append((Vanilla,time_index))
            except:
                print sys.exc_info()
                print "Problem with time index {0}".format(time_index)
                print "Graph ",graph_filename
        #===========================================
        # PARALLELIZATION
        #===========================================

        jobs = []
        for worker_index in range(number_of_workers):
            try:
                print "THIS NAME: ",VANILLA_GRAPHS[worker_index][0]
                p = multiprocessing.Process(target=macro_state_process, args=(gd_directory,
                                                                          macrostates_names,
                                                                          VANILLA_GRAPHS[worker_index][0],
                                                                          dynamics_identifier,
                                                                          macrostates_run_ideintifier,
                                                                          VANILLA_GRAPHS[worker_index][1]))
                jobs.append(p)
                p.start()
            except:
                pass

        current_index += number_of_workers

def macro_state_process(gd_directory,
                        macrostates_names,
                        Vanilla,
                        dynamics_identifier,
                        macrostates_run_ideintifier,
                        time_index):
    """
    """
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
    ALL_TIME_INDEXES.sort()
    #TODO: check if dynamics parameters are complete
    dynamics_identifier = DYNAMICS_PARAMETERS["dynamics_identifier"]
    #TODO: parallelize calls to macrostates
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
                        "u1" : [u1c2, u1c2, .., u1Cm],
                        ....
                        "Un" : [unc2, unc2, .., UnCm]
                    }
    """
    print "Computing bigclam Macro..."
    args = nargs[0]
    bigClamObj = BigClam(Graph, numberOfCommunity=args["number_of_community"], maxNumberOfIterations=args["max_number_of_iterations"])
    # return  { "1JByGBoyCaLcpKdQqKDbJ99vx74owoxUxU": 4.999642500079724,...} = { node_label: Fu1_value, ... }
    return dict(zip(Graph.get_networkx().nodes(), bigClamObj.F.flatten()))
#========================================================================================================================
# THE FOLLOWING DICTIONARY HOLDS ALL MACROSTATES WHICH CAN BE CALLED BY THE EVOLUTION FUNCTION OF GRAPH DYNAMICS
#========================================================================================================================

macrostate_function_dictionary = {
                                  "degree_nodes":degree_nodes,
                                  "node2vec_macrostates":node2vec_macrostates,
                                  "basic_stats":basic_stats,
                                  "pagerank":networkx_pagerank,
                                  "new_nodes":new_nodes,
                                  "bigclam":bigclam,
                                  "deepwalk_online": deepwalk_online,
                                  "node2vec_online_macrostates": node2vec_online_macrostates,
                                  "advanced_stats": advanced_stats,
                                  "degree_centrality":degree_centrality
                                  }
