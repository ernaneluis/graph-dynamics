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

def walks(G, number_of_walks=10, walk_length=10, start_nodes=[]):
    """
    Parameters
    ----------

    G: networkx graph
    number_of_walks: int
    walk_length: int
    start_nodes: list

    Return
    ------
    Walks
    """

    if not start_nodes:
        start_nodes = G.nodes()


    walks =[]
    for i in range(number_of_walks):
        for node in start_nodes:
            walk = [node]
            next_node = node
            while len(walk) < walk_length:
                neigs = G.neighbors(next_node)
                if len(neigs) == 0:
                    break
                next_node = random.choice(neigs)
                walk.append(next_node)
            walks.append(walk)

    return walks
