'''
Created on May 3, 2017

@author: cesar
'''
import json
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause

from graph_dynamics.dynamics.GenerativeDynamics import TxDynamics
from graph_dynamics.networks.tx_graph import TxGraph

class Test(unittest.TestCase):
    
    
    def generateTxGraph(self):

        G = TxGraph(numberOfNodes=15,
                    activity_gamma=2.8,
                    rescaling_factor=10,
                    threshold_min=0.001,
                    delta_t=1)
        #coment this line if you dont want amount/random walker
        G.setAmount(pareto_gama=1.9, threshold=0.001, number_walkers=2)

        dynamics = TxDynamics(initial_graph=G, number_of_connections=2)

        series = dynamics.generate_graphs_series(number_of_steps=5, output_type="t")

        # good for small nodes
        position = nx.shell_layout(G.network)
        # good for big nodes
        # position = nx.random_layout(network.GRAPH)

        for idx, s in enumerate(series):
            self.visualize_graph(s, position=position, time=idx)
            print("number of connections:  " + str(s.number_of_nodes()))
            print("number of active nodes: " + str(len(s.get_active_nodes())))
            pause(3)
            plt.clf()  # Clear figure



    def visualize_graph(self, txgraph, position, time):
        print("##### visualize graph on time " + str(time))
        ## put together a color map, one color for a category
        # print('{}'.format(title) + ' step: {}'.format(T))
        plt.title('step: {}'.format(time))

        # active nodes are red
        # non active nodes are grey
        color_map = {1: 'r', 0: 'grey'}
        # draw
        nx.draw(txgraph.network,
                pos=position,
                with_labels=True,
                node_color=[color_map[txgraph.get_node_type(n)] for n in txgraph.network.nodes()])  ## construct a list of colors then pass to node_color


        # walkers are blue
        if(txgraph.hasGraphAmount):
            print txgraph.get_walkers()
            # add walkers to the plot as blue nodes
            nx.draw_networkx_nodes(txgraph.network, position, node_size=1000, nodelist=txgraph.get_walkers(), node_color='blue')
            # add amount of each node as green color
            labels = {}
            for idx, label in enumerate(txgraph.get_nodes_amounts()):
                labels[idx] = label

            for p in position:  # raise text positions
                t = list(position[p])
                t[1] += 0.20
                position[p] = tuple(t)

            nx.draw_networkx_labels(txgraph.network, position, labels, font_size=10, font_color='g')




if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateTxGraph']
    unittest.main()