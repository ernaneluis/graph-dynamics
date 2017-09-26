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
# from graph_dynamics.communities.bigclam import BigClam
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics import Macrostates

from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.dynamics import FromFilesDynamics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import random
from graph_dynamics.dynamics.GenerativeDynamics import PerraDynamics
from graph_dynamics.networks.datatypes import PerraGraph
import itertools
import operator
from collections import Counter
from joblib import Parallel, delayed
import json
import subprocess
from graph_dynamics.communities.bigclam import BigClam
class Test(unittest.TestCase):

    DYNAMICS_PARAMETERS = {"number_of_steps": 100,
                     "number_of_steps_in_memory": 1,
                     "simulations_directory": "/Users/ernaneluis/Developer/graph-dynamics/simulations/",
                     "dynamics_identifier": "perragraph",
                     "macrostates": [("basic_stats", ())],
                     "graph_class": "PerraGraph",
                     "datetime_timeseries": False,
                     "initial_date": 1,
                     "verbose": True,
                     }


    def define_initial_graph(self):

        #Defines the graph ########################################
        perra_graph_parameters = { "name_string" : "PerraGraph",
                                  "number_of_nodes": 100,
                                   "activity_gamma": 2.8,
                                   "rescaling_factor": 1,
                                   "threshold_min": 0.01,
                                   "delta_t": 1,
                                   "number_walkers": 10,
                                   "graph_state": {"None":None},
                                   "networkx_graph": None # the initial graph: used for empiral data
                                   }



        perra_graph = PerraGraph(identifier_string=perra_graph_parameters["name_string"],
                           graph_state=perra_graph_parameters["graph_state"],
                           networkx_graph=perra_graph_parameters["networkx_graph"],
                           number_of_nodes=perra_graph_parameters["number_of_nodes"],
                           activity_gamma=perra_graph_parameters["activity_gamma"],
                           rescaling_factor=perra_graph_parameters["rescaling_factor"],
                           threshold_min=perra_graph_parameters["threshold_min"],
                           delta_t=perra_graph_parameters["delta_t"],
                           number_walkers=perra_graph_parameters["number_walkers"])

        return perra_graph

    def run_dynamics(self, perra_graph):

        #Defines Dynamics ######################################################

        perra_dynamics = PerraDynamics(initial_graph=perra_graph, number_of_connections=10, DYNAMICAL_PARAMETERS=self.DYNAMICS_PARAMETERS)

        # run dynamics ========================================================================


        perra_dynamics.evolve(N=self.DYNAMICS_PARAMETERS["number_of_steps"],initial_graph=perra_graph)


    def apply_macro(self):


        # Macro States ========================================================================
        self.DYNAMICS_PARAMETERS["macrostates"] = [
            ("basic_stats", ()),
        ]


    def compute(self):

        perra_graph = self.define_initial_graph()

        # nx.draw(perra_graph.get_networkx())
        # plt.show()


        self.run_dynamics(perra_graph)





if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()