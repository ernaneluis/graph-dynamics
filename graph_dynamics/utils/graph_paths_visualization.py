'''
Created on Jul 4, 2017

@author: cesar
'''

import pylab
import matplotlib
import numpy as np
import networkx as nx

from matplotlib.pyplot import pause
from matplotlib import pyplot as plt
from graph_dynamics.dynamics import GraphsFormatsHandlers

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.style.use('seaborn-talk')
colors = []
for a in plt.style.library['bmh']['axes.prop_cycle']:
    colors.append(a["color"])

def whichCommunity(community_membership):
    """
    Parameters
    ----------
        community_membership: dictionary of list {c_1:[],c_2:[],...,}
    Returns
    -------
        which_community: dictionary {node_1:c_1,...,} 
    """
    communities_list = []
    nodes_list = []
    for c in community_membership.keys():
        nodes_in_community = community_membership[c]
        community_value = np.repeat(c,len(nodes_in_community))
        nodes_list.append(nodes_in_community)
        communities_list.append(community_value)
    which_community = dict(zip(np.concatenate(nodes_list),np.concatenate(communities_list)))
    return which_community 

def plotGraphPaths(ax,graph_series,series_name="graph_series",show=True,plot_dir=None,series_type="list",colors=None):
    """
    Simple plot 
    
    Parameters
    
        graph_series: list of networkx objects
        plot_dir:
    """
    padding = 0.1
    # TODO: obtain the full static "collapsed graph" obtain the positions and plot over time from it
    T = len(graph_series)
    static_graph = GraphsFormatsHandlers.staticGraphInducedBySeries(graph_series)
    position = nx.spring_layout(static_graph)
    
    print "Full position for the dynamics ready"
    print "Total number of edges in whole paths history: {0} ".format(static_graph.number_of_edges())
    
    R = np.array(position.values())
    deltaX = abs(max(R[:,0]) - min(R[:,0]))
    deltaY = abs(max(R[:,1]) - min(R[:,1]))
    
    ax.set_title(series_name.format(0))
    ax.set_xlim(min(R[:,0])-padding*deltaX,max(R[:,0])+padding*deltaX)
    ax.set_ylim(min(R[:,1])-padding*deltaY,max(R[:,1])+padding*deltaY)
    nx.draw_networkx(graph_series[0], position,axis=ax,with_labels=False)

    print "#==============================================="
    print "# PRINTING GRAPH SERIES"
    print "# {0} ".format(series_name)
    print "#==============================================="
    
    
    if show:
        pylab.ion()
    for time_index in range(1,T):
        if series_type == "list":
            ax.set_title(series_name.format(time_index))
            nx.draw_networkx(graph_series[time_index], position,axis=ax,with_labels=False)
            if show:
                pause(0.2)
            if plot_dir != None:
                plt.savefig(plot_dir+series_name+"{0}.pdf")
        else:
            print "GRAPH SERIES NOT FORM LIST"
            raise Exception


def plotGraphPathsCommunities(ax,graph_series,community_membership,community_colors,series_name="graph_series",show=True,plot_dir=None,series_type="list",colors=None):
    """
    Simple plot 
    
    Parameters
        ax: matplotlib ax
        
        graph_series: list of networkx objects
        
        plot_dir: string
            where to plot
             
        community_colors: dictionary 
            community and colors {c_1:color,...,}
    """
    padding = 0.1
    # TODO: obtain the full static "collapsed graph" obtain the positions and plot over time from it
    T = len(graph_series)
    static_graph = GraphsFormatsHandlers.staticGraphInducedBySeries(graph_series)
    position = nx.spring_layout(static_graph)
    
    which_community = whichCommunity(community_membership)
    node_color_list = [community_colors[which_community[node]] for node in static_graph.nodes()]
    
    R = np.array(position.values())
    deltaX = abs(max(R[:,0]) - min(R[:,0]))
    deltaY = abs(max(R[:,1]) - min(R[:,1]))
    
    
    ax.set_xlim(min(R[:,0])-padding*deltaX,max(R[:,0])+padding*deltaX)
    ax.set_ylim(min(R[:,1])-padding*deltaY,max(R[:,1])+padding*deltaY)
    
    nx.draw_networkx(graph_series[0], position,axis=ax,node_color=node_color_list,with_labels=False)

    print "#==============================================="
    print "# PRINTING GRAPH SERIES"
    print "# {0} ".format(series_name)
    print "#==============================================="
    
    
    if show:
        pylab.ion()
    for i in range(1,T):
        if series_type == "list":
            nx.draw_networkx(graph_series[i], position,axis=ax,node_color=node_color_list, with_labels=False)
            if show:
                pause(0.2)
            if plot_dir != None:
                plt.savefig(plot_dir+series_name+"{0}.pdf")
        else:
            print "GRAPH SERIES NOT FORM LIST"
            raise Exception