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
from scipy.interpolate import spline
from scipy.interpolate import UnivariateSpline

class Test(unittest.TestCase):



    def convert2temporalmotif(self, output_directory, name):

        # creating temporal graph file input
        input_file = output_directory + name + ".txt"
        path = output_directory + name + ".temporalmotif"
        file = open(path, "w")

        count = 0
        with  open(input_file) as data:
            for row in data:
                line = row.replace("\n", "").split("\t")
                map_to_index_0  = line[1]
                map_to_index_1  = line[2]
                time_index      = line[3]

                file.write(str(map_to_index_0) + " " + str(map_to_index_1) + " " + str(time_index) + "\n")
                count = count + 1
                if count % 10000 == 0:
                    print str(count)

        file.close()
        print "Temporal File: " + path
        return path

    def temporal_motif(self, directory, name):
        # http://snap.stanford.edu/temporal-motifs/code.html

        # with  open(input) as f:
        #     for line in f:
        #         print line
        exe_directory = "/Users/ernaneluis/Developer/graph-dynamics/snap-cpp/examples/temporalmotifs/temporalmotifsmain"
        # file.close()
        input =  directory + name + ".temporalmotif"
        output = directory + name + ".temporalmotifcount"
        args1 = "-i:" + input
        args2 = "-o:" + output
        args3 = "-delta:3600" # 1 hour in secodns
        # calling command of snap in c++
        subprocess.call([exe_directory, args1, args2])
        return output

    def view(self, directory, name):

        input        = directory + name + ".temporalmotifcount"
        output_image = directory + name + "_temporalmotif.png"
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
                ax.text(idx, idy, value, va='center', ha='center', fontsize=10)

        # set tick marks for grid
        ax.set_xticks(np.arange(min_val - diff / 2, max_val - diff / 2))
        ax.set_yticks(np.arange(min_val - diff / 2, max_val - diff / 2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.colorbar(img)

        plt.savefig(output_image)
        print "Image: " + output_image
        plt.show()

    def view_dist(self, directory, name):

        # fig, ax = plt.subplots(1, 1)

        # x  = range(0,36)
        # x_smooth = np.linspace(0, 36, 200)
        for idx in range(0,940):
            name1 = name + "_" + str(idx)
            input        = directory + name1 + ".temporalmotifcount"
            output_image = directory + name1 + "_temporalmotif.png"

            if os.path.exists(input):

                data = np.genfromtxt(input, dtype=None)
                y = data.flatten()
                if sum(y) > 0:
                    print "View: " + input

                    p, x = np.histogram(y, bins=10)
                    x = x[:-1] + (x[1] - x[0]) / 2
                    f = UnivariateSpline(x, p, s=10)
                    plt.plot(x, f(x))

                    # plt.hist(y, bins=10 )
                    # plt.hist(y, histtype='step')
                    # y = np.log2(y)

                    # y_smooth = spline(x, y, x_smooth)
                    # ax.bar(x,y)
                    # ax.plot(x_smooth,y_smooth)
        # plt.yscale('log', nonposy='clip')
        # plt.savefig(output_image)
        print "Image: " + output_image
        plt.show()

    def compute(self):

        dir       = "/Users/ernaneluis/Developer/master_thesis/bigclam/old_bitcoin_dataset/temporal_day/"
        name      = "bitcoin_temporal"


        # # # 2.
        # self.convert2temporalmotif(dir, name)
        # # 3.
        # self.temporal_motif(dir, name)
        # # 4.
        self.view_dist(dir, name)


        print "done"

if __name__ == '__main__':
    import sys;sys.argv = ['', 'Test.compute']
    unittest.main()


 # with  open(input) as f:
        #     for line in f:
        #         print line