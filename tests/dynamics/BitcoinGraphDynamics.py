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

from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.dynamics import FromFilesDynamics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import random
import graph_dynamics.dynamics.GenerativeDynamics as dynamics
import graph_dynamics.networks.datatypes  as graph
import itertools
import operator
from collections import Counter
from joblib import Parallel, delayed
import json
import subprocess
from graph_dynamics.communities.bigclam import BigClam
class Test(unittest.TestCase):

    DYNAMICS_PARAMETERS = {"number_of_steps": 222,
                     "number_of_steps_in_memory": 1,
                     "simulations_directory": "/Users/ernaneluis/Developer/graph-dynamics/simulations/",
                     "dynamics_identifier": "activitydriven",
                     "graph_class": "ActivityDrivenGraph",
                     "datetime_timeseries": False,
                     "initial_date": 1,
                     "verbose": True,
                     }


    temporalmotif_nargs = {
        "delta": 10, # deltas as number of connections
    }

    DYNAMICS_PARAMETERS["macrostates"] = [
        ("basic_stats", ()),
        ("advanced_stats", ()),
        ("degree_centrality", ()),
        ("degree_nodes", ()),
        ("temporalmotif", (temporalmotif_nargs,))
    ]

    ad_dynamics_parameters = {"name_string": "ActivityDrivenGraph",
                        "number_of_nodes": 1000,
                        "activity_gamma": 2, # or 2.8
                        "rescaling_factor": 1,
                        "threshold_min": 0.01,
                        "delta_t": 1,
                        "graph_state": {"None": None},
                        "networkx_graph": None,  # the initial graph: used for empiral data
                        "number_of_connections": 10
                        }


    def define_initial_graph(self):

        #Defines the graph ########################################

        initial_graph = nx.barabasi_albert_graph(self.ad_dynamics_parameters["number_of_nodes"], 3)

        the_graph = graph.ActivityDrivenGraph(graph_state={"None": None},
                                                 networkx_graph=initial_graph)

        return the_graph

    def run_dynamics(self, initial_graph):

        #Defines Dynamics ######################################################

        dynamics_obj = dynamics.ActivityDrivenDynamics(
            initial_graph=initial_graph,
            DYNAMICAL_PARAMETERS=self.DYNAMICS_PARAMETERS,
            extra_parameters= self.ad_dynamics_parameters)

        # run dynamics ========================================================================


        dynamics_obj.evolve(N=self.DYNAMICS_PARAMETERS["number_of_steps"],initial_graph=initial_graph)


    def apply_macro(self):


        # Macro States ========================================================================
        self.DYNAMICS_PARAMETERS["macrostates"] = [
            ("basic_stats", ()),
        ]


    def compute(self):

        ad_graph = self.define_initial_graph()

        # nx.draw(perra_graph.get_networkx())
        # plt.show()


        self.run_dynamics(ad_graph)





if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()