'''
Created on Sep 2, 2017

@author: cesar
'''
from graph_dynamics.dynamics import MacrostatesHandlers, datatypes

def embeddings_time_series(gd_directory):
    """
    """
    for i in range():
        graph_object = datatypes.get_graph_from_dynamics(gd_directory,i)
        
        node_embedding = MacrostatesHandlers.time_index_macro(gd_directory,
                                             "node2vec_online_macrostates",
                                             "node2vec_online",
                                             i)
    
        communities = graph_object.get_graph_state()["communities"]
        new_nodes = MacrostatesHandlers.time_index_macro(gd_directory,
                                                     "new_nodes",
                                                     "newnodes",
                                                     i)