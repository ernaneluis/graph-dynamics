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
import math
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


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


    def view_multiple_temporalmotif(self, data, labels):
        fig = plt.figure(figsize=(14, 21    ))

        rows = len(data)
        columns = 1
        for id, data1 in enumerate(data):

            # Set up Axes
            # rows (1), the number of columns (1) and the plot number (1)
            ax = fig.add_subplot(rows, columns, id + 1)
            self.view_temporalmif_by_time(data1, ax, fig, labels[id])

        plt.subplots_adjust(hspace=0.4)
        # fig.savefig(path + "histogram_" + str(idx) + ".png")
        plt.show()

    def view_temporalmif_by_time(self, data, ax, fig, label):


        # fig, ax = plt.subplots()
        ax.set_yticks(range(1, 37))
        ax.invert_yaxis()
        data_t = np.transpose(data)
        heatmap = ax.pcolor(data_t, cmap="tab20c_r")  # https://matplotlib.org/examples/color/colormaps_reference.html

        plt.title(label)
        plt.xlabel("Time Step as Day")
        plt.ylabel("Patterns")
        fig.colorbar(heatmap)


    def view_error(self, error_x):

        # plt.plot(error_x)
        # plt.yscale('log')
        plt.semilogy(error_x)
        plt.xlabel("Generation")
        plt.ylabel("Error(log)")
        plt.grid(True)
        plt.show()


    def pca(self, data):

        X = np.array(data)

        # result = PCA(X)


        pca = PCA(n_components=36)
        pca.fit(X)
        X_pca = pca.transform(X)

        # for X_transformed in X_pca:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], lw=2)

        # plt.scatter(pca.components_)
        plt.show()

        # a =pca.explained_variance_ratio_
        # print(pca.explained_variance_ratio_)

        # print(pca.singular_values_)
        # The components_ array has shape (n_components, n_features) so components_[i, j] is already giving you the (signed) weights of the contribution of feature j to component i
        # b = pca.components_

        # c= np.abs(pca.components_[0]).argsort()[::-1][:3]

        # x = []
        # y = []
        # z = []
        # for item in result.Y:
        #     x.append(item[0])
        #     y.append(item[1])
        #     z.append(item[2])
        #
        # plt.close('all')  # close all latent plotting windows
        # fig1 = plt.figure()  # Make a plotting figure
        # ax = Axes3D(fig1)  # use the plotting figure to create a Axis3D object.
        # pltData = [x, y, z]
        # ax.scatter(pltData[0], pltData[1], pltData[2], 'bo')  # make a scatter plot of blue dots from the data
        #
        # # make simple, bare axis lines through space:
        # xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0),
        #              (0, 0))  # 2 points make the x-axis line at the data extrema along x-axis
        # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')  # make a red line for the x-axis.
        # yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])),
        #              (0, 0))  # 2 points make the y-axis line at the data extrema along y-axis
        # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')  # make a red line for the y-axis.
        # zAxisLine = ((0, 0), (0, 0), (
        # min(pltData[2]), max(pltData[2])))  # 2 points make the z-axis line at the data extrema along z-axis
        # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')  # make a red line for the z-axis.
        #
        # # label the axes
        # ax.set_xlabel("x-axis label")
        # ax.set_ylabel("y-axis label")
        # ax.set_zlabel("y-axis label")
        # ax.set_title("The title of the plot")
        # plt.show()  # show the plot




        # return pca


    def compute(self):

        # ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = self.get_dynamics_golden()
        ALL_TIME_INDEXES = range(0,222)
        golden_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.golden_gd_directory, self.golden_macrostate_file_indentifier)
        #

        simulation_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.simulation_gd_directory, self.simulation_macrostate_file_indentifier)

        # error_x = self.compute_error_by_time(golden_temporalmotif_by_time, simulation_temporalmotif_by_time)


        # self.view_temporalmif_by_time(golden_temporalmotif_by_time, "Temporal Motif by day(Bitcoin)")
        # self.view_temporalmif_by_time(simulation_temporalmotif_by_time, "Temporal Motif by day(Simulation)")


        # self.view_multiple_temporalmotif([golden_temporalmotif_by_time, simulation_temporalmotif_by_time], ["Temporal Motif by day(Bitcoin)", "Temporal Motif by day(Simulation)"])

        self.pca(golden_temporalmotif_by_time)

if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()