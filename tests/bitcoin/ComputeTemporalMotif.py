'''
Created on July 20, 2017

@author: ernaneluis
'''
import os
# os.system("taskset -p 0xff %d" % os.getpid())
import dill
import operator
import unittest
from itertools import groupby
# import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import re
import pymongo
from pymongo import MongoClient
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import json
import graph_dynamics.dynamics.GenerativeDynamics as dynamics
import graph_dynamics.networks.datatypes  as graph_datatypes
from graph_dynamics.dynamics import Macrostates
import tests.bitcoin.TemporalmotifAnalysisMultiple as analysis_multiple
import time
import sys
from shutil import copyfile
from abc import ABCMeta, abstractmethod
import subprocess
from graph_dynamics.networks.temporalmotif import TemporalMotif


class ComputeTemporalMotif(object):


    def getKey(self,item):
        time = item[2]
        return time

    def simulation_gd_to_temporalmotif(self, input):
        # creating temporal graph file input

        output_path = input.replace(".gd", "") + ".temporalmotif"

        print "Converting graph dynamics to temporalmoitf output_path: " + output_path

        set_nodes = set()
        edges = []
        with open(input, "r") as f:
            for line in f:
                a = line.split(" ")
                from_node = int(a[0])+1
                to_node = int(a[1])+1
                set_nodes.add(from_node)
                set_nodes.add(to_node)
                a.pop(0)
                a.pop(0)
                c = r"".join(a).strip().replace("{","").replace("}","")
                b = c.split(":")
                time = int(b[-1])
                edges.append([from_node,to_node,time])


        if os.path.isfile(output_path) == False:
            output_file = open(output_path, "w")
            edges_sorted = sorted(edges, key=self.getKey)
            for idy, edge in enumerate(edges_sorted):
                output_file.write(str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]) + "\n")
            output_file.close()
        print "Temporal File: " + output_path
        return output_path

    def realdata_gd_to_temporalmotif(self, input):
        # creating temporal graph file input

        output_path = input.replace(".gd", "") + ".temporalmotif"

        print "Converting graph dynamics to temporalmoitf output_path: " + output_path

        set_nodes = set()
        edges = []
        with open(input, "r") as f:
            for line in f:
                a = line.split("\t")
                from_node = a[0]
                to_node = a[1]
                set_nodes.add(from_node)
                set_nodes.add(to_node)
                a.pop(0)
                a.pop(0)
                c = r"".join(a).strip().replace("{","").replace("}","")
                b = c.split(":")
                time = int(b[-1])
                edges.append([from_node,to_node,time])


        if os.path.isfile(output_path) == False:
            output_file = open(output_path, "w")

            indexes = range(1, len(set_nodes)+1)
            all_nodes = dict(zip(set_nodes, indexes))
            edges_sorted = sorted(edges, key=self.getKey)

            for idy, edge in enumerate(edges_sorted):
                froma = all_nodes.get(edge[0])
                toa = all_nodes.get(edge[1])
                time = edge[2]
                output_file.write(str(froma) + " " + str(toa) + " " + str(time) + "\n")
            output_file.close()
        print "Temporal File: " + output_path
        return output_path

    def temporal_motif(self, input, delta):

        exe_directory = "../../snap-cpp/examples/temporalmotifs/temporalmotifsmain"  # path of excecutable

        output_motif = input.replace(".temporalmotif","") + ".temporalmotifcount"

        if os.path.isfile(output_motif ) == False:

            args1 = "-i:" + input
            args2 = "-o:" + output_motif
            args3 = "-delta:" + str(delta)

            # calling command of snap in c++
            subprocess.call([exe_directory, args1, args2, args3])

    def compute_temporalmotif_from_simulation_gd(self, input, delta):

        gd_file = input

        temporalmotif_file = self.simulation_gd_to_temporalmotif(gd_file)

        self.temporal_motif(temporalmotif_file, delta)

    def compute_temporalmotif_from_realdata_gd(self, input, delta):

        gd_file = input

        temporalmotif_file = self.realdata_gd_to_temporalmotif(gd_file)

        self.temporal_motif(temporalmotif_file, delta)


if __name__ == '__main__':

    # input = "/Volumes/Ernane/simulations/activitydriven1515247743001_gd/activitydriven1515247743001_gGD_0_.gd"
    comp = ComputeTemporalMotif()
    # comp.compute_temporalmotif_from_realdata_gd(realdata, 3600)
    # comp.compute_temporalmotif_from_simulation_gd(realdata, 3600)

    for idx in range(0,222):
        input = "/Volumes/Ernane/day_gd/day_gGD_"+str(idx)+"_.gd"
        comp.compute_temporalmotif_from_realdata_gd(input, 3600)