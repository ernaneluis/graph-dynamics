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
from PlotMotifs import PlotMotifs
from ComputeTemporalMotif import ComputeTemporalMotif
from operator import itemgetter
import copy
class SimulationDynamics(object):

    __metaclass__ = ABCMeta
    def __init__(self, DYNAMICS_PARAMETERS, temporalmotif_nargs, MACROSTATES_PARAMETERS, model_dynamics_parameters):

        self.DYNAMICS_PARAMETERS = DYNAMICS_PARAMETERS
        self.temporalmotif_nargs = temporalmotif_nargs
        self.MACROSTATES_PARAMETERS = MACROSTATES_PARAMETERS
        self.model_dynamics_parameters = model_dynamics_parameters

        self.gd_dir = DYNAMICS_PARAMETERS["simulations_directory"] + DYNAMICS_PARAMETERS[
            "dynamics_identifier"] + "_gd/"
        self.gd_name = DYNAMICS_PARAMETERS["dynamics_identifier"]


    def save_parameters(self):

        gd_dir = self.DYNAMICS_PARAMETERS["simulations_directory"] + self.DYNAMICS_PARAMETERS["dynamics_identifier"] + "_gd/"

        with open(gd_dir + 'SIMULATIONS_PARAMETERS.json', 'w') as fp:
            json.dump(self.model_dynamics_parameters, fp)

        with open(gd_dir + 'MACROSTATES_PARAMETERS.json', 'w') as fp2:
            json.dump(self.MACROSTATES_PARAMETERS, fp2)

    @abstractmethod
    def define_initial_graph(self):
        raise NotImplemented()

    @abstractmethod
    def run_dynamics(self, initial_graph):
        raise NotImplemented()

    def evolve_dynamics(self, dynamics_obj, initial_graph):
        # run dynamics ========================================================================
        dynamics_obj.evolve(N=self.DYNAMICS_PARAMETERS["number_of_steps"], initial_graph=initial_graph)
        self.save_parameters()

    def apply_macro(self):
        # compute macros
        Macrostates.evaluate_vanilla_macrostates(self.gd_dir, self.MACROSTATES_PARAMETERS, self.gd_name)

    def compress_gd(self):

        time_indexes = map(int, [filename.split("_")[2] for filename in os.listdir(self.gd_dir) if "_gGD_" in filename])
        time_indexes = sorted(time_indexes)

        final_file = self.gd_dir + self.gd_name + "_gGD_0_.gd"

        filenames = [self.gd_dir + self.gd_name + "_gGD_" + str(idx) + "_.gd" for idx in time_indexes]
        with open(final_file, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

        for fname in filenames[1:]:
            os.remove(fname)

        print final_file
        return final_file

class SimulationBitcoinMemoryDynamics(SimulationDynamics):


    def __init__(self, index=None):
        if(index == None):
            index = str(int(time.time()))

        DYNAMICS_PARAMETERS = {"number_of_steps": 100,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": "/Volumes/Ernane/final-data/simulations/",
                               "dynamics_identifier": "memoryallBigclam"+index,
                               "graph_class": "BitcoinGraph",
                               "datetime_timeseries": False,
                               "initial_date": 0,
                               "verbose": True,
                               "macrostates": []
                               }

        temporalmotif_nargs = {
            "delta": 3600,  # deltas as 1h  in seconds
        }

        MACROSTATES_PARAMETERS = [
            ("basic_stats", ()),
            # ("basic_stats", ()),
            # ("advanced_stats", ()),
            # ("degree_centrality", ()),
            # ("degree_nodes", ()),
            # ("temporalmotif", (temporalmotif_nargs,))
        ]

        model_dynamics_parameters = {
            "name_string": "BitcoinMemoryGraph",
                                "number_of_nodes": 2000,
                                "number_of_connections": 10,  # max number of connection a node can make
                                "memory_queue_size": 5,

            "activity_gamma": 2,  # or 2.8
            "activity_rescaling_factor": 1, # avr number of active nodes per unit of time
            "activity_threshold_min": 0.0001,
            "activity_delta_t": 1,

            "memory_number_of_connections": 25,

            "memory_activity_gamma": 2,  # or 2.8
            "memory_activity_rescaling_factor": 1,  # avr number of active nodes per unit of time
            "memory_activity_threshold_min": 0.0001,
            "memory_activity_delta_t": 1,

            "graph_state": {"None": None},
            "networkx_graph": None,  # the initial graph: used for empiral data
            "delta_in_seconds": 3600,
            "amount_pareto_gama": 2.8,
            "amount_threshold": 0.0001,
            "activity_transfer_function": "x/math.sqrt(1+pow(x,2))",
            "number_new_nodes": 1
        }


        SimulationDynamics.__init__(self, DYNAMICS_PARAMETERS, temporalmotif_nargs, MACROSTATES_PARAMETERS, model_dynamics_parameters)

    def define_initial_graph(self):
        #Defines the graph ====================================================================
        initial_graph  = nx.Graph()
        initial_graph.add_nodes_from(list(xrange(self.model_dynamics_parameters["number_of_nodes"])))

        max_number_of_nodes = self.model_dynamics_parameters["number_of_nodes"] \
                              + ((self.DYNAMICS_PARAMETERS["number_of_steps"] - 1)
                                 * self.model_dynamics_parameters["number_new_nodes"])

        # memory = [list() for n in self.model_dynamics_parameters["number_of_nodes"]]
        # graph_state = json.dumps(memory)
        #
        #

        gstate = {"None": None}

        btg = graph_datatypes.BitcoinMemoryGraph(graph_state=gstate,
                                           networkx_graph=initial_graph)
        # btg.number_of_connections =
        btg.memory_size = self.model_dynamics_parameters["memory_queue_size"]

        btg.init_amount(max_number_of_nodes ,self.model_dynamics_parameters["amount_pareto_gama"], self.model_dynamics_parameters["amount_threshold"])

        # ==================  set up the initial activity potential  =======================================
        btg.init_activity_potential(max_number_of_nodes,
                                    self.model_dynamics_parameters["activity_gamma"],
                                    self.model_dynamics_parameters["activity_threshold_min"],
                                    self.model_dynamics_parameters["activity_rescaling_factor"],
                                    self.model_dynamics_parameters["activity_delta_t"])

        # ==================  set up the initial memory activity potential  =======================================
        btg.init_memory_activity_potential(max_number_of_nodes,
                                    self.model_dynamics_parameters["memory_activity_gamma"],
                                    self.model_dynamics_parameters["memory_activity_threshold_min"],
                                    self.model_dynamics_parameters["memory_activity_rescaling_factor"],
                                    self.model_dynamics_parameters["memory_activity_delta_t"])

        btg.set_nodes_active()
        btg.set_nodes_memory_active()

        btg.set_connections(number_of_connections=self.model_dynamics_parameters["number_of_connections"],
                            delta_in_seconds=self.model_dynamics_parameters["delta_in_seconds"], time_step=0)



        btg.update_graph_state()

        return btg

    def run_dynamics(self, initial_graph):

        # defines Dynamics ====================================================================
        dynamics_obj = dynamics.BitcoinMemoryDynamics(
            initial_graph=initial_graph,
            DYNAMICAL_PARAMETERS=self.DYNAMICS_PARAMETERS,
            extra_parameters= self.model_dynamics_parameters)

        self.evolve_dynamics(dynamics_obj, initial_graph)

    def getKey(self,item):
        time = item[2]
        return time


    def compute(self):

        self.run_dynamics(self.define_initial_graph())

        gd_file = self.compress_gd()

        self.apply_macro()

        return gd_file



if __name__ == '__main__':



    time = str(int(time.time())) + "00"

    ## production
    simulation = SimulationBitcoinMemoryDynamics(time)
    comp_motif = ComputeTemporalMotif()
    plot_motif = PlotMotifs()

    gd_file = simulation.compute()


    temporalmotif_file = comp_motif.compute_temporalmotif_from_simulation_gd(gd_file, simulation.temporalmotif_nargs["delta"])


    datas = ["/Volumes/Ernane/final-data/golden/daymodel122_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/final-data/golden/daymodel165_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/final-data/golden/daymodel210_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/final-data/simulations/activitydriven151563604500_gd/activitydriven151563604500_gGD_0_.temporalmotifcount",
             temporalmotif_file]

    labels = ["Day A", "Day B", "Day C", "Activity Driven", "Simulation"]

    output = "/Volumes/Ernane/final-data/simulations/" + simulation.DYNAMICS_PARAMETERS["dynamics_identifier"] + "_gd/" + "all_motifs_golden_activity_vs_simulation_" + simulation.DYNAMICS_PARAMETERS["dynamics_identifier"]

    output_norm = "/Volumes/Ernane/final-data/simulations/" + simulation.DYNAMICS_PARAMETERS["dynamics_identifier"] + "_gd/" + "all_motifs_golden_activity_vs_simulation_norm_" + simulation.DYNAMICS_PARAMETERS["dynamics_identifier"]

    columns_all = range(0, 36)

    columns_relevant = [0, 1, 4, 5, 6, 10, 11, 14, 15,  18, 20, 21, 27, 29, 32, 35]
    data_relevant = copy.deepcopy(datas)
    labels_relevant = copy.deepcopy(labels)

    data_relevant_relative = copy.deepcopy(datas)
    labels_relevant_relative = copy.deepcopy(labels)

    columns_cycles = [1, 4, 11, 14, 21, 27, 29]
    data_cycles = copy.deepcopy(datas)
    labels_cycles = copy.deepcopy(labels)

    data_cycles_relative = copy.deepcopy(datas)
    labels_cycles_relative = copy.deepcopy(labels)

    plot_motif.plot_bar(data_relevant, labels_relevant, columns_relevant, output_norm + "_relevant", normalize=1)
    plot_motif.plot_bar(data_cycles, labels_cycles, columns_cycles, output_norm + "_cycles", normalize=1)
    plot_motif.plot_bar(datas, labels, columns_all, output, normalize=0)
    plot_motif.plot_bar(datas, labels, columns_all, output_norm, normalize=1)

    plot_motif.plot_bar(data_relevant_relative, labels_relevant_relative, columns_relevant, output_norm + "_relevant_relative", normalize=1)
    plot_motif.plot_bar(data_cycles_relative, labels_cycles_relative, columns_cycles, output_norm + "_cycles_relative", normalize=2)

# m11(star) 0
# m12(cycle) 1
# m15(cycle) 4
# m16(star) 5
# m21(star)6
# m25(star) 10
# m26(cycle)11
# m33(cycle)14
# m34(star)15
# m41(star)18
# m43(star)20
# m44(cycle)21
# m54(cycle),27
# m56(cycle),29
# m63(star) 32
# m66(star)35