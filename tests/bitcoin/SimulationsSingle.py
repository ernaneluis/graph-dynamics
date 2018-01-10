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

class SimulationActivityDrivenDynamics(SimulationDynamics):


    def __init__(self, index=None):

        DYNAMICS_PARAMETERS = {"number_of_steps": 24,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": "/Volumes/Ernane/simulations/",
                               "dynamics_identifier": "activitydriven"+index,
                               "graph_class": "ActivityDrivenGraph",
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
            ("temporalmotif", (temporalmotif_nargs,))
        ]

        model_dynamics_parameters = {"name_string": "ActivityDrivenGraph",
                                  "number_of_nodes": 1000,
                                  "activity_gamma": 2,  # or 2.8
                                  "rescaling_factor": 1,
                                  "threshold_min": 0.0001,
                                  "delta_t": 1,
                                  "graph_state": {"None": None},
                                  "networkx_graph": None,  # the initial graph: used for empiral data
                                  "number_of_connections": 1,  # max number of connection a node can make
                                  "delta_in_seconds": 3600
                                  }


        SimulationDynamics.__init__(self, DYNAMICS_PARAMETERS, temporalmotif_nargs, MACROSTATES_PARAMETERS, model_dynamics_parameters)

    def define_initial_graph(self):

        #Defines the graph ====================================================================

        # initial_graph = nx.barabasi_albert_graph(self.ad_dynamics_parameters["number_of_nodes"], 3)
        initial_graph  = nx.Graph()
        initial_graph.add_nodes_from(list(xrange(self.model_dynamics_parameters["number_of_nodes"])))

        return graph_datatypes.ActivityDrivenGraph(graph_state={"None": None}, networkx_graph=initial_graph)


    def run_dynamics(self, initial_graph):

        # defines Dynamics ====================================================================
        dynamics_obj = dynamics.ActivityDrivenDynamics(
            initial_graph=initial_graph,
            DYNAMICAL_PARAMETERS=self.DYNAMICS_PARAMETERS,
            extra_parameters= self.model_dynamics_parameters)

        self.evolve_dynamics(dynamics_obj, initial_graph)

    def compute(self):

        gd_dir = self.DYNAMICS_PARAMETERS["simulations_directory"] + self.DYNAMICS_PARAMETERS["dynamics_identifier"] + "_gd/"
        gd_name = self.DYNAMICS_PARAMETERS["dynamics_identifier"]

        self.run_dynamics(self.define_initial_graph())

        self.compress_gd()

        self.apply_macro()


class SimulationBitcoinDynamics(SimulationDynamics):


    def __init__(self):
        DYNAMICS_PARAMETERS = {"number_of_steps": 24,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": "/Volumes/Ernane/simulations/",
                               "dynamics_identifier": "bitcoinmodelmemoryto8",
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
            ("temporalmotif", (temporalmotif_nargs,))
        ]

        model_dynamics_parameters = {
            "name_string": "BitcoinGraph",
            "number_of_nodes": 1000,
            "activity_gamma": 2,  # or 2.8
            "activity_rescaling_factor": 1, # avr number of active nodes per unit of time
            "activity_threshold_min": 0.0001,
            "activity_delta_t": 1,
            "graph_state": {"None": None},
            "networkx_graph": None,  # the initial graph: used for empiral data
            "number_of_connections": 10,  # max number of connection a node can make
            "delta_in_seconds": 3600,
            "number_walkers": 100,
            "amount_pareto_gama": 2.8,
            "amount_threshold": 0.0001,
            "activity_transfer_function": "x/math.sqrt(1+pow(x,2))",
            "number_new_nodes": 5
        }

        self.gd_dir = DYNAMICS_PARAMETERS["simulations_directory"] + DYNAMICS_PARAMETERS[
            "dynamics_identifier"] + "_gd/"
        self.gd_name = DYNAMICS_PARAMETERS["dynamics_identifier"]

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

        btg = graph_datatypes.BitcoinGraph(graph_state=gstate,
                                           networkx_graph=initial_graph)
        # btg.number_of_connections =

        btg.init_amount(max_number_of_nodes ,self.model_dynamics_parameters["amount_pareto_gama"], self.model_dynamics_parameters["amount_threshold"])

        # ==================  set up the initial activity potential  =======================================
        btg.init_activity_potential(max_number_of_nodes,
                                    self.model_dynamics_parameters["activity_gamma"],
                                    self.model_dynamics_parameters["activity_threshold_min"],
                                    self.model_dynamics_parameters["activity_rescaling_factor"],
                                    self.model_dynamics_parameters["activity_delta_t"])

        btg.set_nodes_active()
        btg.set_connections(number_of_connections=self.model_dynamics_parameters["number_of_connections"],
                            delta_in_seconds=self.model_dynamics_parameters["delta_in_seconds"])
        btg.update_graph_state()

        # gstate = json_graph.node_link_data(btg)
        # gstate = gstate["nodes"]

        return btg


    def run_dynamics(self, initial_graph):

        # defines Dynamics ====================================================================
        dynamics_obj = dynamics.BitcoinDynamics(
            initial_graph=initial_graph,
            DYNAMICAL_PARAMETERS=self.DYNAMICS_PARAMETERS,
            extra_parameters= self.model_dynamics_parameters)

        self.evolve_dynamics(dynamics_obj, initial_graph)

    def compute(self):



        self.run_dynamics(self.define_initial_graph())

        self.compress_gd(self.gd_dir, self.gd_name)

        self.apply_macro(self.gd_dir, self.gd_name, self.MACROSTATES_PARAMETERS)


class SimulationBitcoinMemoryDynamics(SimulationDynamics):


    def __init__(self, index=None):
        if(index == None):
            index = str(int(time.time()))

        DYNAMICS_PARAMETERS = {"number_of_steps": 100,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": "/Volumes/Ernane/simulations/",
                               "dynamics_identifier": "newmemorymodel"+index,
                               "graph_class": "BitcoinGraph",
                               "datetime_timeseries": False,
                               "initial_date": 0,
                               "verbose": True,
                               "macrostates": []
                               }

        self.temporalmotif_nargs = {
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
                                "number_of_nodes": 100,
            "activity_gamma": 2,  # or 2.8
            "activity_rescaling_factor": 1, # avr number of active nodes per unit of time
            "activity_threshold_min": 0.0001,
            "activity_delta_t": 1,
                                "number_of_connections": 1,  # max number of connection a node can make
            "memory_activity_gamma": 2,  # or 2.8
            "memory_activity_rescaling_factor": 1,  # avr number of active nodes per unit of time
            "memory_activity_threshold_min": 0.0001,
            "memory_activity_delta_t": 1,
            "memory_number_of_connections": 2,
                                "memory_queue_size": 5,
            "graph_state": {"None": None},
            "networkx_graph": None,  # the initial graph: used for empiral data
            "delta_in_seconds": 3600,
            "amount_pareto_gama": 2.8,
            "amount_threshold": 0.0001,
            "activity_transfer_function": "x/math.sqrt(1+pow(x,2))",
            "number_new_nodes": 2
        }


        SimulationDynamics.__init__(self, DYNAMICS_PARAMETERS, self.temporalmotif_nargs, MACROSTATES_PARAMETERS, model_dynamics_parameters)

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

    def gd_to_temporalmotif(self, input):
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

    def temporal_motif(self, input, delta):

        exe_directory = "../../snap-cpp/examples/temporalmotifs/temporalmotifsmain"  # path of excecutable

        output_motif = input.replace(".temporalmotif","") + ".temporalmotifcount"

        if os.path.isfile(output_motif ) == False:

            args1 = "-i:" + input
            args2 = "-o:" + output_motif
            args3 = "-delta:" + str(delta)

            # calling command of snap in c++
            subprocess.call([exe_directory, args1, args2, args3])

    def compute(self):

        self.run_dynamics(self.define_initial_graph())

        gd_file = self.compress_gd()

        temporalmotif_file = self.gd_to_temporalmotif(gd_file)

        self.temporal_motif(temporalmotif_file, self.temporalmotif_nargs["delta"])

        # self.apply_macro()


class MarcoFromGD(SimulationDynamics):


    def __init__(self, dir, name):
        DYNAMICS_PARAMETERS = {"number_of_steps": 24,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": dir,
                               "dynamics_identifier": name,
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
            ("temporalmotif", (temporalmotif_nargs,))
        ]

        SimulationDynamics.__init__(self, DYNAMICS_PARAMETERS, temporalmotif_nargs, MACROSTATES_PARAMETERS, {})

        # self.apply_macro()

    def define_initial_graph(self):
        a = 1

    def run_dynamics(self, initial_graph):
        a = 1


if __name__ == '__main__':

    # macro = MarcoFromGD("/Volumes/Ernane/simulations/", "daymodel165")
    # macro.apply_macro()



    # simulation = SimulationBitcoinDynamics(time + str(idx))
    # simulation.compute()


    time = str(int(time.time())) + "00"
    last_path = ""
    all_paths_all_cycle = []
    all_paths_all_non_zero = []
    all_paths_all_relevant = []
    for idx in range(1,2,1):
        print "running: " + str(idx)
        #
        # simulation      = SimulationActivityDrivenDynamics(time + str(idx))
        # simulation.compute()

        ## production
        simulation = SimulationBitcoinMemoryDynamics(time+str(idx))
        simulation.compute()

        simulation_gd_directory = simulation.gd_dir
        simulation_macrostate_file_indentifier = simulation.gd_name

        ## test
        # simulation_gd_directory = "/Volumes/Ernane/simulations/simpleymemory1514937263_gd/"
        # simulation_macrostate_file_indentifier = "simpleymemory1514937263"

        # comparing to golden model
        golden_gd_directory = ["/Volumes/Ernane/simulations/daymodel122_gd/",
                               "/Volumes/Ernane/simulations/daymodel165_gd/",
                               "/Volumes/Ernane/simulations/daymodel210_gd/",
                               "/Volumes/Ernane/simulations/activitydriven1515247743001_gd/"]

        golden_macrostate_file_indentifier = ["daymodel122", "daymodel165", "daymodel210", "activitydriven1515247743001"]

        ALL_TIME_INDEXES = range(0,1)


        # sys.stdout = open(simulation_gd_directory+"result.txt", "w")


        analysis = analysis_multiple.TemporalmotifAnalysisMultiple(golden_gd_directory, golden_macrostate_file_indentifier,
                                                                   simulation_gd_directory,
                                                                   simulation_macrostate_file_indentifier, ALL_TIME_INDEXES)


        datas = [analysis.golden_temporalmotif_by_time[0],
             analysis.golden_temporalmotif_by_time[1],
             analysis.golden_temporalmotif_by_time[2],
             analysis.golden_temporalmotif_by_time[3],
             analysis.simulation_temporalmotif_by_time  ]
        labels = ["Day A", "Day B", "Day C", "Activity Driven", "Simulation"]

        headers_cycle = ['Model', 'M15', 'M26', 'M33', 'M44', 'M53', 'M54', 'M55', 'M56', 'error']

        # [1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 17, 19, 21, 22, 23, 26, 27, 28, 29]
        headers_all_non_zero = ['Model','M12','M13', 'M14','M15','M22', 'M23','M26','M31','M32',
                                'M33','M36','M42',
                                'M44','M45', 'M46',
                                'M53', 'M54', 'M55', 'M56', 'error']

        headers_all_relevant = ['Model','M13', 'M14','M15', 'M23','M26',
                                'M33','M36',
                                'M44','M45', 'M46',
                                'M53', 'M54', 'M55', 'M56', 'error']

        results_all_cycle       = analysis.results_csv(datas, labels, 'cycle', headers_cycle, simulation_macrostate_file_indentifier + '_results_all_cycle')
        results_all_non_zero    = analysis.results_csv(datas, labels, 'all', headers_all_non_zero, simulation_macrostate_file_indentifier + '_results_all_non_zero')
        results_all_relevant    = analysis.results_csv(datas, labels, 'relevant', headers_all_relevant, simulation_macrostate_file_indentifier + '_results_all_relevant')

        #

        analysis.plot_bar(results_all_cycle, simulation_macrostate_file_indentifier    + "_results_all_cycle")
        analysis.plot_bar(results_all_non_zero, simulation_macrostate_file_indentifier + "_results_all_non_zero")
        analysis.plot_bar(results_all_relevant, simulation_macrostate_file_indentifier + "_results_all_relevant")

        last_path = simulation_gd_directory

        all_paths_all_cycle.append(simulation_gd_directory + simulation_macrostate_file_indentifier    + "_results_all_cycle.csv")

        all_paths_all_non_zero.append(simulation_gd_directory + simulation_macrostate_file_indentifier + "_results_all_non_zero.csv")

        all_paths_all_relevant.append(simulation_gd_directory + simulation_macrostate_file_indentifier + "_results_all_relevant.csv")


        # sys.stdout.close()

    analysis.merge_csv(all_paths_all_cycle,last_path + "merge_all_cycles.csv")
    analysis.merge_csv(all_paths_all_non_zero, last_path + "merge_all_non_zero.csv")
    analysis.merge_csv(all_paths_all_relevant, last_path + "merge_all_relevant.csv")