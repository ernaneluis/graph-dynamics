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
import pandas as pd
from joblib import Parallel, delayed

def convert_egdelist2bigclam(gd_directory, name):

    time_indexes, series_graph = load_graph_dynamics(gd_directory, name)
    # creating bitclam graph file input
    for idx, graph in enumerate(series_graph):
        path = gd_directory  + name + "_bigclam_" + str(time_indexes[idx]) + ".txt"
        file = open(path, "w")
        nodes = list(set(graph.nodes()))

        nodes = sorted(nodes)
        indexes = range(0, len(nodes))
        all_nodes = dict(zip(nodes, indexes))

        print path + " egdes: "+ str(len(graph.edges()))
        for idy, edge in enumerate(graph.edges()):
            map_to_index_0 = all_nodes.get(edge[0])
            map_to_index_1 = all_nodes.get(edge[1])
            file.write(str(map_to_index_0) + "\t" + str(map_to_index_1) + "\n")
        file.close()

    print "done convert"

def getKey(item):
    time = item[0]
    return time

def computeBigClam(exe_directory, input):
    # http://snap.stanford.edu/temporal-motifs/code.html

    # time_indexes = map(int, [filename.split("_")[3].replace(".txt", "") for filename in os.listdir(gd_directory) if "_bigclam_" in filename])

    # for idx in time_indexes:

    output = input.replace(".txt","")

    args1 = "-i:" + input

    #  -c:The number of communities to detect (-1: detect automatically) (default:-1)
    args3 = "-c:1000"
    # -mc:Minimum number of communities to try (default:5)
    args4 = "-mc:1"
    # -xc:Maximum number of communities to try (default:100)
    args5 = "-xc:10"
    # -nc:How many trials for the number of communities (default:10)
    args6 = "-nc:1"
    # -nt:Number of threads for parallelization (default:1) -nt:1 means no parallelization.
    args7 = "-nt:2"

    args2 = "-o:" + output + args3.replace(":","-")

    # calling command of snap in c++ args4,
    subprocess.call([exe_directory, args1, args2, args3, args7])


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

def convert_bigclam_sorted(input):
    df = pd.read_csv(input, delimiter="\n")
    print df
    a = df.values
    correct = [item[0].split("\t") for item in a]
    edges_sorted = sorted(correct, key=getKey)

    file = open(input.replace(".txt", ".bigclam"), "w")

    print "convert_egdelist2bigclam: "

    for idy, edge in enumerate(edges_sorted):
        to_n = int(edge[0]) + 1
        from_n = int(edge[1])+1
        file.write(str(to_n) + "\t" + str(from_n) + "\n")
    file.close()


def convert_temporal_to_bigclam(input):
    df = pd.read_csv(input, delimiter="\n")
    # print df
    values = df.values
    edges = []
    for item in values:
        sp = item[0].split("\t")[0]
        spau = sp.split(" ")
        edges.append([spau[0],spau[1]])

    # edges_sorted = sorted(correct, key=getKey)
    output = input.replace(".temporalmotif", ".bigclam")
    file = open(output, "w")

    print "convert_egdelist2bigclam: "

    for idy, edge in enumerate(edges):
        to_n = int(edge[0])
        from_n = int(edge[1])
        file.write(str(to_n) + "\t" + str(from_n) + "\n")
    file.close()
    return output

def compute(i):
    exe_path = "../../snap-cpp/examples/bigclam/bigclam"
    input = "/Volumes/Ernane/bigclam/tocompute/day_bigclam_" + str(i) + ".txt"
    computeBigClam(exe_path, input)

if __name__ == '__main__':

    #path of excecutable
    exe_path = "/Volumes/Ernane/bigclam/snap-master/examples/bigclam/bigclam"

    # name            = "graph_bigclam"
    # 1.
    # reads gd files and covnert to biglcam input format
    # convert_egdelist2bigclam(path, name)
    # 2.



    # 3.
    # time_indexes, series = load_bigclam(path, name)
    # 4.    # Save Figure
    # visualization_histogram(series, path)

    # convert_bigclam_sorted("/Volumes/Ernane/bigclam/daymodel122_gGD_0_.bigclam")

    # n_jobs
    # The maximum number of concurrently running jobs, such as the number of Python worker processes when backend=multiprocessing
    # or the size of the thread-pool when backend=threading. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
    # which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
    # https://pythonhosted.org/joblib/generated/joblib.Parallel.html

    # Parallel(n_jobs=6)(delayed(compute)(i) for i in range(3,101))

    input = "/Volumes/Ernane/bigclam/memoryallBigclam151575059300_gGD_0_.temporalmotif"
    input_converted = convert_temporal_to_bigclam(input)

    exe_path = "../../snap-cpp/examples/bigclam/bigclam"
    computeBigClam(exe_path, input_converted)

    print "Done"