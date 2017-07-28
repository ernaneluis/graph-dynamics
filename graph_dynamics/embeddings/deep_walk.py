'''
Created on Jul 25, 2017

@author: cesar
'''
import numpy as np
import networkx as nx
import random


def deepWalk(G,number_of_walks=10):
    """
    Parameters
    ----------
    
    G: networkx graph
    number_of_walks: int
    
    Return
    ------
    Walks
    """
    walks =[]
    for i in range(number_of_walks):
        for node in G.nodes():
            walk = [node]
            next_node = node
            for j in range(number_of_walks):
                next_node = random.choice(G.neighbors(next_node))
                walk.append(next_node)
            walks.append(walk)
    
    walks = [map(str, walk) for walk in walks]
    return walks
