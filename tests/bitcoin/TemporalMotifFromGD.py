'''
Created on July 20, 2017

@author: ernaneluis
'''
import os
import sys
import networkx as nx
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import sys;
import unittest
import bisect
class Test(unittest.TestCase):


    def load_graph_dynamics(self, path, name):

        time_indexes = map(int,[filename.split("_")[2] for filename in os.listdir(path) if "_gGD_" in filename])
        series_graph = []
        time_indexes = sorted(time_indexes)
        for idx in time_indexes:
            fname = path + name + "_gGD_" + str(idx) + "_.gd"
            print "Reading: " + fname
            networkx_graph = nx.read_edgelist(fname)
            series_graph.append(networkx_graph)

        return time_indexes, series_graph

    def convert_gd2temporal(self, gd_directory, time_indexes, series_graph):
        # creating temporal graph file input
        path = gd_directory + "temporal-graph.txt"
        file = open(path, "w")
        set_nodes = set()

        for idx in time_indexes:
            graph = series_graph[idx]
            nodes = set(graph.nodes())
            set_nodes = set_nodes.union(nodes)

        indexes = range(0,len(set_nodes))
        all_nodes = dict(zip(set_nodes, indexes))
        # all_nodes = list(set_nodes)
        for idx in time_indexes:
            print "Converting graph #: " + str(idx)
            graph = series_graph[idx]

            total = len(graph.edges())
            for idy, edge in enumerate(graph.edges()):

                map_to_index_0 = all_nodes.get(edge[0])
                map_to_index_1 = all_nodes.get(edge[1])

                file.write(str(map_to_index_0) + " " + str(map_to_index_1) + " " + str(idx + 1) + "\n")

                if idy % 100 == 0:
                    print str(idy) + "/" + str(total)

        file.close()
        print "Temporal File: " + path
        return path

    def temporal_motif(self, input, output_dir):
        # http://snap.stanford.edu/temporal-motifs/code.html
        exe_directory = "/Users/ernaneluis/Developer/graph-dynamics/snap-cpp/examples/temporalmotifs/temporalmotifsmain"

        output = output_dir + "temporal-graph-counts.txt"
        args1 = "-i:" + input
        args2 = "-o:" + output
        args3 = "-delta:300"
        # calling command of snap in c++
        subprocess.call([exe_directory, args1, args2, args3])
        return output

    def view(self, input, output_dir):
        print "View: " + input
        data = np.genfromtxt(input, dtype=None)
        # plt.matshow(data, cmap='Blues', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        fig, ax = plt.subplots()
        min_val = 0
        max_val = 6
        diff = 1

        img = ax.imshow(data, cmap='Blues', interpolation='nearest')

        for idx, row in enumerate(data.transpose()):
            for idy, value in enumerate(row):
                ax.text(idx, idy, value, va='center', ha='center')

        # set tick marks for grid
        ax.set_xticks(np.arange(min_val - diff / 2, max_val - diff / 2))
        ax.set_yticks(np.arange(min_val - diff / 2, max_val - diff / 2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.colorbar(img)
        image = output_dir + "temporal_motif.png"
        plt.savefig(image)
        print "Image: " + image
        plt.show()

    def compute(self):

        path            = "/Users/ernaneluis/Developer/master_thesis/bigclam/bitcoin/"
        name            = "bitcoin"

        # # 1.
        time_indexes, series_graph = self.load_graph_dynamics(path, name)
        # # 2.
        file_path       = self.convert_gd2temporal(path, time_indexes, series_graph)
        # # 3.
        temporal_count  = self.temporal_motif(file_path, path)
        # # 4Z.
        self.view(temporal_count, path)

        print "done"

if __name__ == '__main__':
    import sys;sys.argv = ['', 'Test.compute']
    unittest.main()
