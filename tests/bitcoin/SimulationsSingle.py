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

from abc import ABCMeta, abstractmethod

class SimulationDynamics(object):

    __metaclass__ = ABCMeta
    def __init__(self, DYNAMICS_PARAMETERS, temporalmotif_nargs, MACROSTATES_PARAMETERS, model_dynamics_parameters):

        self.DYNAMICS_PARAMETERS = DYNAMICS_PARAMETERS
        self.temporalmotif_nargs = temporalmotif_nargs
        self.MACROSTATES_PARAMETERS = MACROSTATES_PARAMETERS
        self.model_dynamics_parameters = model_dynamics_parameters


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

    def apply_macro(self, gd_directory, gd_name, macro_params ):
        # compute macros
        Macrostates.evaluate_vanilla_macrostates(gd_directory, macro_params, gd_name)

    def compress_gd(self, gd_directory, gd_name):

        time_indexes = map(int, [filename.split("_")[2] for filename in os.listdir(gd_directory) if "_gGD_" in filename])
        time_indexes = sorted(time_indexes)

        final_file = gd_directory + gd_name + "_gGD_0_.gd"

        filenames = [gd_directory + gd_name + "_gGD_" + str(idx) + "_.gd" for idx in time_indexes]
        with open(final_file, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

        for fname in filenames[1:]:
            os.remove(fname)

        print final_file

class SimulationActivityDrivenDynamics(SimulationDynamics):


    def __init__(self):

        DYNAMICS_PARAMETERS = {"number_of_steps": 24,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": "/Volumes/Ernane/simulations/",
                               "dynamics_identifier": "nullmodelltest",
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
                                  "number_of_nodes": 10000,
                                  "activity_gamma": 2,  # or 2.8
                                  "rescaling_factor": 1,
                                  "threshold_min": 0.0001,
                                  "delta_t": 1,
                                  "graph_state": {"None": None},
                                  "networkx_graph": None,  # the initial graph: used for empiral data
                                  "number_of_connections": 100,  # max number of connection a node can make
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

        self.compress_gd(gd_dir, gd_name)

        self.apply_macro(gd_dir, gd_name, self.MACROSTATES_PARAMETERS )


class SimulationPerraDynamics(SimulationDynamics):


    def __init__(self):
        DYNAMICS_PARAMETERS = {"number_of_steps": 24,
                               "number_of_steps_in_memory": 1,
                               "simulations_directory": "/Volumes/Ernane/simulations/",
                               "dynamics_identifier": "perramodel",
                               "graph_class": "PerraGraph",
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
            "name_string": "PerraGraph",
            "number_of_nodes": 10000,
            "activity_gamma": 2,  # or 2.8
            "rescaling_factor": 1,
            "threshold_min": 0.0001,
            "delta_t": 1,
            "graph_state": {"None": None},
            "networkx_graph": None,  # the initial graph: used for empiral data
            "number_of_connections": 100,  # max number of connection a node can make
            "delta_in_seconds": 3600,
            "number_walkers": 100
        }


        SimulationDynamics.__init__(self, DYNAMICS_PARAMETERS, temporalmotif_nargs, MACROSTATES_PARAMETERS, model_dynamics_parameters)

    def define_initial_graph(self):
        #Defines the graph ====================================================================
        initial_graph  = nx.Graph()
        initial_graph.add_nodes_from(list(xrange(self.model_dynamics_parameters["number_of_nodes"])))
        return graph_datatypes.PerraGraph(graph_state={"None": None}, networkx_graph=initial_graph)


    def run_dynamics(self, initial_graph):

        # defines Dynamics ====================================================================
        dynamics_obj = dynamics.PerraDynamics(
            initial_graph=initial_graph,
            DYNAMICAL_PARAMETERS=self.DYNAMICS_PARAMETERS,
            extra_parameters= self.model_dynamics_parameters)

        self.evolve_dynamics(dynamics_obj, initial_graph)

    def compute(self):

        gd_dir = self.DYNAMICS_PARAMETERS["simulations_directory"] + self.DYNAMICS_PARAMETERS["dynamics_identifier"] + "_gd/"
        gd_name = self.DYNAMICS_PARAMETERS["dynamics_identifier"]

        self.run_dynamics(self.define_initial_graph())

        self.compress_gd(gd_dir, gd_name)

        self.apply_macro(gd_dir, gd_name, self.MACROSTATES_PARAMETERS )

if __name__ == '__main__':

    # simulation      = SimulationActivityDrivenDynamics()
    # simulation.compute()

    simulation2 = SimulationPerraDynamics()
    simulation2.compute()