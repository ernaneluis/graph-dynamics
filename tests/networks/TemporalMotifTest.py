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

class Test(unittest.TestCase):



    def convert_gd2temporal(self, input_file, output_directory):

        set_times = set()
        total = 0

        with  open(input_file) as f:
            for row in f:
                line = row.replace("\n","").split("\t")
                set_times.add(line[2])
                total = total + 1

        all_times = list(set_times)
        all_times = sorted(all_times)

        indexes = range(0, len(all_times))
        all_keys = dict(zip(all_times, indexes))

        # creating temporal graph file input
        path = output_directory + "lt_temporal-graph.txt"
        file = open(path, "w")

        count = 0
        with  open(input_file) as data:
            for row in data:
                line = row.replace("\n", "").split("\t")
                map_to_index_0 = line[0]
                map_to_index_1 = line[1]

                time_index = all_keys.get(line[2])

                file.write(str(map_to_index_0) + " " + str(map_to_index_1) + " " + str(time_index+1) + "\n")
                count = count + 1
                if count % 10000 == 0:
                    print str(count) + "/" + str(total)

        file.close()
        print "Temporal File: " + path
        return path

    def temporal_motif(self, input, output_dir):
        # http://snap.stanford.edu/temporal-motifs/code.html

        # with  open(input) as f:
        #     for line in f:
        #         print line
        exe_directory = "/Users/ernaneluis/Developer/graph-dynamics/snap-cpp/examples/temporalmotifs/temporalmotifsmain"
        # file.close()

        output = output_dir + "lt_temporal-graph-counts.txt"
        args1 = "-i:" + input
        args2 = "-o:" + output
        args3 = "-delta:300"
        # calling command of snap in c++
        subprocess.call([exe_directory, args1, args2])
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

        output_dir       = "/Users/ernaneluis/Developer/master_thesis/bigclam/bitcoin_graphs/"
        input_file       = "/Users/ernaneluis/Developer/master_thesis/bigclam/bitcoin_graphs/lt_graph.txt"
        # name            = "bitcoin"


        # # # 2.
        file_path       = self.convert_gd2temporal(input_file, output_dir)
        # # 3.
        temporal_count  = self.temporal_motif(file_path, output_dir)
        # # 4.
        self.view(temporal_count, output_dir)

        print "done"

if __name__ == '__main__':
    import sys;sys.argv = ['', 'Test.compute']
    unittest.main()


 # with  open(input) as f:
        #     for line in f:
        #         print line