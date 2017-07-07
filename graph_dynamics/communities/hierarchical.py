'''
Created on Jun 13, 2017

@author: cesar
'''
import numpy as np
from tag2hierarchy import hierarchy
from tag2hierarchy.hierarchy import treeHandlers

def communitiesPerHierarchyLevel(HMM):
    """
    Parameters:
        HMM: HIerarchical Graph Class
    Returns:
        levelsPartition: dictionary per level with each community assignment
    """
    nodesPerLevel = treeHandlers.obtainNodesPerLevel(HMM.hierarchy)
    levels = nodesPerLevel.keys()
    levelsPartition = {level:{k:0 for k in nodesPerLevel[level]} for level in levels}
    for level in levels:
        nodes_in_level = nodesPerLevel[level]
        for node in nodes_in_level:
            levelsPartition[level][node] = treeHandlers.obtainNodeCargo(HMM.hierarchy,node)['NodesInCommunity']
    return levelsPartition

