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


def convert_egdelist2bigclam(gd_directory, name):

    time_indexes, series_graph = load_graph_dynamics(gd_directory, name)
    # creating bitclam graph file input
    for idx, graph in enumerate(series_graph):
        path = gd_directory + "_" + name + "_" + str(time_indexes[idx]) + ".txt"
        file = open(path, "w")
        nodes = list(set(graph.nodes()))
        print path + " egdes: "+ str(len(graph.edges()))
        for idy, edge in enumerate(graph.edges()):
            map_to_index_0 = nodes.index(edge[0])
            map_to_index_1 = nodes.index(edge[1])
            file.write(str(map_to_index_0) + "\t" + str(map_to_index_1) + "\n")
        file.close()

    print "done convert"


def computeBigClam(exe_directory, gd_directory, name):
    # http://snap.stanford.edu/temporal-motifs/code.html

    time_indexes = map(int, [filename.split("_")[2].replace(".txt", "") for filename in os.listdir(gd_directory) if "_bigclam_" in filename])

    for idx in time_indexes:
        input = gd_directory + name  + "_" + str(idx) + ".txt"
        output = gd_directory + name + "_" + str(idx) + "_"

        args1 = "-i:" + input
        args2 = "-o:" + output
        args3 = "-c:1"

        # calling command of snap in c++
        subprocess.call([exe_directory, args1, args2, args3])


def load_graph_dynamics(path, name):
    time_indexes = map(int,[filename.split("_")[2] for filename in os.listdir(path) if "_gGD_" in filename])
    graphs       = []


    for idx in time_indexes:
        fname = path + name + "_gGD_" + str(idx) + "_.gd"
        fh = open(fname, 'rb')
        networkx_graph = nx.read_edgelist(fh)
        graphs.append(networkx_graph)
        fh.close()

    return time_indexes, graphs


def load_bigclam(path, name):

    time_indexes = map(int,[filename.split("_")[2] for filename in os.listdir(path) if "_f_matrix" in filename])
    time_indexes = sorted(time_indexes)
    matrices       = []

    for idx in time_indexes:
        fname = path + name + "_" + str(idx) + "_f_matrix.txt"
        if os.path.isfile(fname) == True:
            print fname
            data = np.loadtxt(fname)
            matrices.append(data)

    return time_indexes, matrices

   # Save Figure
def visualization_histogram(series, path):

    series = np.array(series)

    count = 0
    split_series = np.array_split(series, 107)
    for idx, split in enumerate(split_series):

        # max_value   = max(map(lambda x: x[0], split))
        # norm_series = [i/max_value for i in split]
        # leng        = len(norm_series)

        leng        = len(split)

        # Create a Figure
        fig = plt.figure(figsize=(14, 12), dpi=120)

        for id, data in enumerate(split):
            columns = 4
            rows = math.ceil(leng / columns) + 1

            # Set up Axes
            # rows (1), the number of columns (1) and the plot number (1)
            ax = fig.add_subplot(rows, columns, id + 1)

            ax.hist(np.log2(data), log=True, bins=10)

            ax.set_title('Graph ' + str(count + 1))
            # ax.set_xlabel("Valuez")
            ax.set_ylabel("Frequency")
            count = count + 1

        plt.subplots_adjust(hspace=0.8, wspace=0.6)
        fig.savefig(path+"histogram_" + str(idx) + ".png")
    # plt.show()


if __name__ == '__main__':

    exe_path   = "../../snap-cpp/examples/bigclam/bigclam"#path of excecutable
    path            = "../../data/bigclam/"
    name            = "easy"
    # 1.
    #convert_egdelist2bigclam(path, name)
    # 2.
    name            = "easy_bigclam"
    computeBigClam(exe_path, path, name)
    
    # 3.
    #time_indexes, series = load_bigclam(path, name)
    # 4.    # Save Figure
    #visualization_histogram(series, path)

    print "Done"