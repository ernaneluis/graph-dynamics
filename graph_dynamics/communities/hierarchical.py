'''
Created on Jun 13, 2017

@author: cesar
'''
from tag2hierarchy import hierarchy
import numpy as np

def hierarchicalErdosRenyi(numberOfNodes,hierarchy):
    """
    
    Parameters:
    
    numberOfNodes: int     
    hierarchy: tag2hierarchy tree object
    """
    numberOfNodesInTree = len(hierarchy.treeHandlers.nodeNames(hierarchy))
    nodesPerLevel = hierarchy.treeHandlers.obtainNodesPerLevel(hierarchy)
    
    return None