'''
Created on Jul 25, 2017

@author: cesar
'''
import numpy as np
from matplotlib import pyplot as plt
from graph_dynamics.dynamics import MacrostatesHandlers

def all_new_nodes_per_time(gd_directory,initial_nodes,latest_index=10):
    """
    Asummes that macro file is newnodes, and macro name is new_nodes
    sort is performed to guarantee proper alignements
    
    Return
    ------
    
    new_nodes_per_time
    """
    initial_nodes.sort()
    new_nodes_per_time = []
    for time_index in range(1,latest_index):
        new_nodes_per_time.append(MacrostatesHandlers.time_index_macro(gd_directory,
                                                                       "new_nodes",
                                                                       'newnodes',
                                                                        time_index)["new_nodes"])
    for n in new_nodes_per_time:
        n.sort()
    new_nodes_per_time = [initial_nodes] + new_nodes_per_time
    
    return new_nodes_per_time

def embedings_in_order(new_nodes_per_time,
                       time_index_base,
                       time_index_other,
                       gd_directory,
                       macrostate_file_indentifier):
    """
    This function works only for aggregated networks were nodes are not deleted over time, 
    it obtains who are the new nodes every time, it ,,
    
    
    new_nodes_per_time: list of new nodes per time
    
    """
    macro_state_identifier = 'node2vec_macrostates'
    nodes_base = []
    for i in range(0,time_index_base+1):
        nodes_base.extend(new_nodes_per_time[i])
        
    nodes_other = []
    for i in range(0,time_index_other+1):
        nodes_other.extend(new_nodes_per_time[i])
    
    w_base = []
    node_embeddings_base = MacrostatesHandlers.time_index_macro(gd_directory,
                                                                macro_state_identifier,
                                                                macrostate_file_indentifier,
                                                                time_index_base)
    
    node_embeddings_other = MacrostatesHandlers.time_index_macro(gd_directory,
                                                                 macro_state_identifier,
                                                                 macrostate_file_indentifier,
                                                                 time_index_other)
    
    w_base = []
    for node in nodes_base:
        w_base.append(np.array(node_embeddings_base[node]))
    w_base = np.array(w_base)
    
    w_other = []
    for node in nodes_other:
        w_other.append(np.array(node_embeddings_other[node]))
    w_other = np.array(w_other)
    
    
    return w_base, w_other

def procrustes_align(base_embed, other_embed):
    """ 
        Align other embedding to base embeddings via Procrustes.
        Returns best distance-preserving aligned version of other_embed
        NOTE: Assumes indices are aligned
        
        Returns
        -------
        
    """
    rows_base = base_embed.shape[0]
    rows_other = other_embed.shape[0]
    rows_to_align = min(rows_base,rows_other)
    
    basevecs0 = base_embed - base_embed.mean(0)
    othervecs0 = other_embed - other_embed.mean(0)
    
    basevecs = basevecs0[:rows_to_align]
    othervecs = othervecs0[:rows_to_align]
    
    #obtains the rotation
    m = othervecs.T.dot(basevecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    
    #rotates the older vectors
    fixedvecs = othervecs0.dot(ortho)
    fixedvecs = fixedvecs - fixedvecs.mean(0)
    
    return basevecs0,  fixedvecs

def plot_w(a,b):
    plt.scatter(a[:,0], a[:,1],color="red")
    plt.scatter(b[:,0], b[:,1],color="blue")