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
from graph_dynamics.dynamics import Macrostates


def getKey(item):
    return item[2]

def convert_gd2temporal(gd_directory, name, idx):
    # creating temporal graph file input

    print "Converting to temporal #: " + str(idx)

    input_path = gd_directory + name + "_gGD_" + str(idx) + "_.gd"
    output_path = gd_directory + name + "_" + str(idx) + ".temporalmotif"


    input_file = open(input_path, 'rb')
    # //parse_edgelist
    graph       = nx.read_edgelist(input_file)
    total       = len(graph.edges())
    set_nodes   = set(graph.nodes())
    indexes     = range(0, len(set_nodes))
    all_nodes   = dict(zip(set_nodes, indexes))

    output_file = open(output_path, "w")

    edges = graph.edges(data=True)
    edges = sorted(edges, key=getKey)

    for idy, edge in enumerate(edges):

        map_to_index_0 = all_nodes.get(edge[0])
        map_to_index_1 = all_nodes.get(edge[1])
        time           = edge[2]['time']

        output_file.write(str(map_to_index_0) + " " + str(map_to_index_1) + " " + str(time) + "\n")

        if idy % 100 == 0:
            print str(idy) + "/" + str(total)

    output_file.close()
    print "Temporal File: " + output_path
    return output_path


def convert_csv2gd(gd_directory, name, idx):

    print "Converting to GD #: " + str(idx)

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
                if len(line) == 5:
                    map_to_index_0 = line[0]
                    map_to_index_1 = line[1]
                    time           = line[3]

                    # if count_line > 64170:
                    #     print row

                    if map_to_index_0 != '' and map_to_index_1 !='':
                        line = str(map_to_index_0) + "\t" + str(map_to_index_1) + "\t{'time':" + time + "}" + "\n"
                        file.write(line)
    file.close()
    return output_file


def compute_temporal_motif(exe_directory, gd_directory, name, delta, idx):
    # http://snap.stanford.edu/temporal-motifs/code.html
    input  = gd_directory + name  + "_" + str(idx) + ".temporalmotif"
    output = gd_directory + name  + "_" + str(idx) + ".temporalmotifcount"

    args1 = "-i:" + input
    args2 = "-o:" + output
    args3 = "-delta:" + str(delta)

    # calling command of snap in c++
    subprocess.call([exe_directory, args1, args2, args3])


def compute_macros(gd_directory):

    macrostates_run_ideintifier = "day"

    temporalmotif_nargs = {
        "delta": 3600,
    }

    macrostates_names = [
        ("basic_stats", ()),
        ("advanced_stats", ()),
        ("temporalmotif", (temporalmotif_nargs,))
    ]
    # compute macros
    Macrostates.evaluate_vanilla_macrostates(gd_directory, macrostates_names,macrostates_run_ideintifier)

    # Macrostates.evaluate_vanilla_macrostates_parallel(gd_directory, macrostates_names, macrostates_run_ideintifier,4)

def compute(exe_directory, gd_directory, name, delta, idx):

    convert_csv2gd(gd_directory, name, idx)

    convert_gd2temporal(gd_directory, name, idx)

    compute_temporal_motif(exe_directory, gd_directory, name, delta, idx)



if __name__ == '__main__':

    exe_path   = "../../snap-cpp/examples/temporalmotifs/temporalmotifsmain" #path of excecutable
    # gd_directory    = "/Volumes/Ernane/ErnaneEdges/"
    gd_directory = "/Volumes/Ernane/day_gd/"
    name       = "day"
    delta      = 3600 # one hour delta

    # for filename in os.listdir(gd_path):
    #     if "csv" in filename:
    #         print filename
    #         a = int(filename.split("_")[1].replace(".csv", ""))
    #         print a

    # 1 read graph dynamics files
    time_indexes = map(int, [filename.split("_")[1].replace(".csv","") for filename in os.listdir(gd_directory) if "csv" in filename])
    time_indexes = sorted(time_indexes)
    # graphs = []

    # for idx in time_indexes:
    #     compute(exe_path, gd_path, name, idx)

    # n_jobs
    # The maximum number of concurrently running jobs, such as the number of Python worker processes when backend=multiprocessing
    # or the size of the thread-pool when backend=threading. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
    # which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
    # https://pythonhosted.org/joblib/generated/joblib.Parallel.html

    # computeTemporalMotif(exe_path, gd_path, name, delta, 0)

    # Parallel(n_jobs=4) (delayed(compute)(exe_path, gd_directory, name, delta, idx ) for idx in time_indexes)

    #
    # for idx in time_indexes:
    #     compute(exe_path, gd_path, name, delta, idx)\


    compute_macros(gd_directory)


    print "Done"