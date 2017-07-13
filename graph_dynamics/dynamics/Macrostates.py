'''
Created on Jul 12, 2017

@author: cesar

THIS ARE THE FUNCTIONS TO BE 
CALLED BY THE DYNAMICS

ALL FUNCTION SHERE DEFINED MUST RETURN A JSON
'''
import numpy as np
import networkx as nx

def degree_distribution(Graph):
    """
    Parameters:
    ----------
        Graph: graph object    
    """
    return Graph.get_networkx().degree()
    
    
macrostate_function_dictionary = {"degree_distribution":degree_distribution}