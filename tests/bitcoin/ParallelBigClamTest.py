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
from joblib import Parallel, delayed

def convert_csv2gd(gd_directory, name, idx):
    count_line = 0

    output_file = gd_directory + name + "_gGD_" + str(idx) + "_.gd"
    file = open(output_file, "w")

    input_file = gd_directory + name + "_" + str(idx) + ".csv"
    with  open(input_file) as data:
        for row in data:
            # jump the first line of the csv
            count_line = count_line + 1
            if count_line > 1:
                line = row.replace("\n", "").split("\t")
                map_to_index_0 = line[0]
                map_to_index_1 = line[1]

                file.write(str(map_to_index_0) + "\t" + str(map_to_index_1) + "\n")
    file.close()
    return output_file

def convert_gd2bigclam(gd_directory, name, graph, idx):

    nodes = list(set(graph.nodes()))
    nodes = sorted(nodes)
    indexes = range(0, len(nodes))
    all_nodes = dict(zip(nodes, indexes))

    path = gd_directory + name + "_bigclam_" + str(idx) + ".txt"
    file = open(path, "w")

    print "convert_egdelist2bigclam: " + path + " egdes: " + str(len(graph.edges()))

    for idy, edge in enumerate(graph.edges()):
        map_to_index_0 = all_nodes.get(edge[0])
        map_to_index_1 = all_nodes.get(edge[1])
        file.write(str(map_to_index_0) + "\t" + str(map_to_index_1) + "\n")
    file.close()

    print "done convert"
    return path

def computeBigClam(exe_directory, gd_directory, name, idx):
    # http://snap.stanford.edu/temporal-motifs/code.html

    input = gd_directory + name  + "_bigclam_" + str(idx) + ".txt"
    output = gd_directory + name + "_bigclam_" + str(idx) + "_"

    args1 = "-i:" + input
    args2 = "-o:" + output
    args3 = "-c:1"

    # calling command of snap in c++
    subprocess.call([exe_directory, args1, args2, args3])

def compute(exe_path, gd_path, name, idx):

    fname = convert_csv2gd(gd_path, name, idx)

    # fname = gd_path + name + "_gGD_" + str(idx) + "_.gd"
    fh = open(fname, 'rb')
    print "Read GD: " + fname

    networkx_graph = nx.read_edgelist(fh)
    # graphs.append(networkx_graph)
    fh.close()

    path_bigclam_input = convert_gd2bigclam(gd_path, name, networkx_graph, idx)

    computeBigClam(exe_path, gd_path, name, idx)

if __name__ == '__main__':

    exe_path   = "../../snap-cpp/examples/bigclam/bigclam"#path of excecutable
    gd_path    = "../../data/bigclam/"
    name       = "easy"

    # 1 read graph dynamics files
    time_indexes = map(int, [filename.split("_")[1].replace(".csv","") for filename in os.listdir(gd_path) if "csv" in filename])
    # graphs = []

    # for idx in time_indexes:
    #     compute(exe_path, gd_path, name, idx)

    # n_jobs
    # The maximum number of concurrently running jobs, such as the number of Python worker processes when backend=multiprocessing
    # or the size of the thread-pool when backend=threading. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
    # which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
    # https://pythonhosted.org/joblib/generated/joblib.Parallel.html

    Parallel(n_jobs=1) (delayed(compute)(exe_path, gd_path, name, idx ) for idx in time_indexes)

    print "Done"