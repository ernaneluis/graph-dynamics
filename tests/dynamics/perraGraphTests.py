'''
Created on May 3, 2017

@author: cesar
'''
import unittest
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.path as mpath


from graph_dynamics.dynamics.GenerativeDynamics import PerraDynamics
from graph_dynamics.networks.perra_graph import PerraGraph

class Test(unittest.TestCase):
    
    
    def generateTxGraph(self):

        G = PerraGraph(numberOfNodes=15,
                    activity_gamma=2.8,
                    rescaling_factor=10,
                    threshold_min=0.001,
                    delta_t=1,
                    number_walkers=2)
        # G.set_walker(number_walkers=2)
        dynamics = PerraDynamics(initial_graph=G, number_of_connections=2)

        series = dynamics.generate_graphs_paths(number_of_steps=5, output_type="t")

        # good for small nodes
        position = nx.shell_layout(G.networkx_graph)
        # good for big nodes
        # position = nx.random_layout(networkx_graph.GRAPH)

        # self.visualize_graph_3D(series=series, position=position)
        self.visualize_graph(series=series, position=position)

        plt.show()



    def visualize_graph(self, series, position):

        for time, txgraph in enumerate(series):
            print("##### visualize graph on time " + str(time))

            # plt.figure()
            ## put together a color map, one color for a category
            # print('{}'.format(title) + ' step: {}'.format(T))
            plt.title('step: {}'.format(time))

            # active nodes are red
            # non active nodes are grey
            color_map = {1: 'r', 0: 'grey'}
            # draw
            nx.draw(txgraph.networkx_graph,
                    pos=position,
                    with_labels=True,
                    node_color=[color_map[txgraph.get_node_type(n)] for n in txgraph.networkx_graph.nodes()])  ## construct a list of colors then pass to node_color


            # walkers are blue
            print txgraph.get_walkers()
            # add walkers to the plot as blue nodes
            nx.draw_networkx_nodes(txgraph.networkx_graph, position, node_size=1000, nodelist=txgraph.get_walkers(), node_color='blue')


            pause(3)
            plt.clf()  # Clear figure

    # https://matplotlib.org/examples/shapes_and_collections/path_patch_demo.html
    # https://matplotlib.org/examples/mplot3d/bars3d_demo.html
    def visualize_graph_3D(self, series, position):

        fig = plt.figure()
        ax = Axes3D(fig)

        for time, txgraph in enumerate(series):
            print("time:  " + str(time))
            # self.visualize_graph_3D(s, position=position, time=idx, ax=ax)

            Path = mpath.Path

            # //add nodes
            for idx, pos in enumerate(position):
                x = position[idx][0]
                y = position[idx][1]

                path_data = [
                    (Path.MOVETO, (x, y)),
                ]
                codes, verts = zip(*path_data)
                path = mpath.Path(verts, codes)

                x_point, y_point = zip(*path.vertices)
                # ax.plot(x_point, y_point, 'go')
                ax.plot(x_point, [time * 10] * len(x_point), 'go', zs=y_point, zdir='z')

            # add edges

            edges = txgraph.networkx_graph.edges()

            for idx, e in enumerate(edges):
                edge = edges[idx]
                node_to = edge[0]
                node_from = edge[1]
                pos_node_to = position[node_to]
                pos_node_from = position[node_from]
                # print(edge)

                edge_path_data = [
                    (Path.MOVETO, (pos_node_to[0], pos_node_to[1])),
                    (Path.MOVETO, (pos_node_from[0], pos_node_from[1])),
                ]

                edge_codes, edge_verts = zip(*edge_path_data)
                path = mpath.Path(edge_verts, edge_codes)

                edge_x_point, edge_y_point = zip(*path.vertices)
                # put 0s on the y-axis, and put the y axis on the z-axis
                # ax.plot(xs=edge_x_point, ys=[0] * len(edge_x_point), zs=edge_y_point, zdir='z', label='ys=0, zdir=z', color='r')
                ax.plot(edge_x_point, [time * 10] * len(edge_x_point), 'r', zs=edge_y_point, zdir='z')


        ax.set_xlabel('X')
        ax.set_ylabel('Time')
        ax.set_zlabel('Y')
        plt.show()
         # 2d
        # fig, ax = plt.subplots()






if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateTxGraph']
    unittest.main()