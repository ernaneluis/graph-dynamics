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
from matplotlib import pyplot as plt
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics import Macrostates

class Test(unittest.TestCase):
    # golden standard
    golden_gd_directory = "/Volumes/Ernane/day_gd/"
    golden_macrostate_file_indentifier = "day"

    simulation_gd_directory  = "/Users/ernaneluis/Developer/graph-dynamics/simulations/activitydriven_gd/"
    simulation_macrostate_file_indentifier = "activitydriven-macros"

    def normalize(self, predictions, targets):
        if sum(predictions) > 0:
            predictions_norm = predictions / np.linalg.norm(predictions)
        else:
            predictions_norm = predictions

        if sum(targets) > 0:
            targets_norm = targets / np.linalg.norm(targets)
        else:
            targets_norm = targets

        return predictions_norm, targets_norm

    # Root mean square error
    def error1(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def get_dynamics_golden(self):
        return gd_files_handler.gd_folder_stats(self.golden_gd_directory, True)

    def get_dynamics_simulation(self):
        return gd_files_handler.gd_folder_stats(self.simulation_gd_directory, True)

    def temporalmotif_by_time(self, ALL_TIME_INDEXES, gd_directory, macrostate_file_identifier):

        temporalmotif_by_time = []
        for i in ALL_TIME_INDEXES:
            dict = MacrostatesHandlers.time_index_macro(gd_directory,
                                                        macro_state_identifier="temporalmotif",
                                                        macrostate_file_indentifier=macrostate_file_identifier,
                                                        time_index=i)
            temporalmotif_by_time.append(dict)

        # array of vectors of 36 parameters
        return temporalmotif_by_time

    def compute_error_by_time(self, golden_temporalmotif_by_time, simulation_temporalmotif_by_time):

        timeline = []
        for idx, row in enumerate(golden_temporalmotif_by_time):
            golden_temporalmotif = golden_temporalmotif_by_time[idx]
            simulation_temporalmotif = simulation_temporalmotif_by_time[idx]

            simulation_temporalmotif_norm, golden_temporalmotif_norm = self.normalize(simulation_temporalmotif, golden_temporalmotif)
            error_i = self.error1(simulation_temporalmotif_norm, golden_temporalmotif_norm)
            timeline.append(error_i)


        return timeline
    def compute(self):

        # ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = self.get_dynamics_golden()
        ALL_TIME_INDEXES = range(0,222)
        golden_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.golden_gd_directory,
                                                                  self.golden_macrostate_file_indentifier)

        simulation_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.simulation_gd_directory,
                                                                      self.simulation_macrostate_file_indentifier)


        error_x = self.compute_error_by_time(golden_temporalmotif_by_time, simulation_temporalmotif_by_time)


        # plt.plot(error_x)
        # plt.yscale('log')
        plt.semilogy(error_x)
        plt.xlabel("Generation")
        plt.ylabel("Error(log)")
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()