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
from graph_dynamics.dynamics import GraphPathsHandlers

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

def plotGraphPaths(graph_series,series_name="graph_series",show=True,plot_dir=None,series_type="list",):
    """
    Simple plot 
    
    Parameters
    
        graph_series: list of networkx objects
        plot_dir:
    """
    padding = 0.1
    # TODO: obtain the full static "collapsed graph" obtain the positions and plot over time from it
    T = len(graph_series)
    static_graph = GraphPathsHandlers.staticGraphInducedBySeries(graph_series)
    position = nx.spring_layout(static_graph)
    
    R = np.array(position.values())
    deltaX = abs(max(R[:,0]) - min(R[:,0]))
    deltaY = abs(max(R[:,1]) - min(R[:,1]))
    
    fig, ax = plt.subplots(1,1,figsize=(12, 14))
    ax.set_xlim(min(R[:,0])-padding*deltaX,max(R[:,0])+padding*deltaX)
    ax.set_ylim(min(R[:,1])-padding*deltaY,max(R[:,1])+padding*deltaY)
    nx.draw_networkx(graph_series[0], position,axis=ax)

    print "#==============================================="
    print "# PRINTING GRAPH SERIES"
    print "# {0} ".format(series_name)
    print "#==============================================="
    
    
    if show:
        pylab.ion()
    for i in range(1,T):
        if series_type == "list":
            nx.draw_networkx(graph_series[i], position,axis=ax)
            if show:
                pause(1)
            if plot_dir != None:
                plt.savefig(plot_dir+series_name+"{0}.pdf")
        else:
            print "GRAPH SERIES NOT FORM LIST"
            raise Exception
