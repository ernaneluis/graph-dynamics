'''
Created on July 20, 2017

@author: ernaneluis
'''
import unittest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics import Macrostates
import math
from sklearn.decomposition import PCA
import pylab
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm

class Test(unittest.TestCase):
    # golden standard
    golden_gd_directory = "/Volumes/Ernane/simulations/daymodel122_gd/"
    golden_macrostate_file_indentifier = "daymodel122"

    # simulation_gd_directory  = "/Volumes/Ernane/simulations/nullmodel24h_gd/"
    # simulation_macrostate_file_indentifier = "nullmodel24h"

    simulation_gd_directory  = "/Volumes/Ernane/simulations/perramodel_gd/"
    simulation_macrostate_file_indentifier = "perramodel"



    motifslabels = ["$M_{1,1}$", "$M_{1,2}$", "$M_{1,3}$", "$M_{1,4}$", "$M_{1,5}$", "$M_{1,6}$",
                    "$M_{2,1}$", "$M_{2,2}$", "$M_{2,3}$", "$M_{2,4}$", "$M_{2,5}$", "$M_{2,6}$",
                    "$M_{3,1}$", "$M_{3,2}$", "$M_{3,3}$", "$M_{3,4}$", "$M_{3,5}$", "$M_{3,6}$",
                    "$M_{4,1}$", "$M_{4,2}$", "$M_{4,3}$", "$M_{4,4}$", "$M_{4,5}$", "$M_{4,6}$",
                    "$M_{5,1}$", "$M_{5,2}$", "$M_{5,3}$", "$M_{5,4}$", "$M_{5,5}$", "$M_{5,6}$",
                    "$M_{6,1}$", "$M_{6,2}$", "$M_{6,3}$", "$M_{6,4}$", "$M_{6,5}$", "$M_{6,6}$"]

    def normalize(self, data):
        if sum(data) > 0:
            return np.array(data) / np.float(np.array(data).max()) # return data / np.linalg.norm(data)
        else:
            return data
        #TODO: check on temporalmotif paper if they count redudency, because if they do only norm by the max

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

            golden_temporalmotif_norm = self.normalize(golden_temporalmotif)
            simulation_temporalmotif_norm = self.normalize(simulation_temporalmotif)
            error_i = self.error1(simulation_temporalmotif_norm, golden_temporalmotif_norm)
            timeline.append(error_i)


        return timeline

    def normalize_series(self, series):
        norm_series = []
        for idx, data in enumerate(series):
            data_norm = self.normalize(data)
            norm_series.append(data_norm)
        return norm_series

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

        data_norm = self.normalize_series(data)

        # fig, ax = plt.subplots()
        ax.set_yticks(range(1, 37))
        ax.invert_yaxis()
        data_t = np.transpose(data_norm)
        heatmap = ax.pcolor(data_t, cmap="tab20c_r")  # https://matplotlib.org/examples/color/colormaps_reference.html

        plt.title(label)
        plt.xlabel("Time Step as Day")
        plt.ylabel("Patterns")
        fig.colorbar(heatmap)
    # TODO add correct normalization
    def view_error(self, error_x):

        # plt.plot(error_x)
        # plt.yscale('log')
        plt.semilogy(error_x)
        plt.xlabel("Generation")
        plt.ylabel("Error(log)")
        plt.grid(True)
        plt.show()

    def view_multiple_bar(self, data, labels):

        fig = plt.figure(figsize=(31, 14))
        ax = fig.add_subplot(111)

        data_norm1 = self.normalize(data[0])
        data_norm2 = self.normalize(data[1])

        w = 0.4
        x1 = np.array(range(1,37))
        x2 = x1 + w

        plt.bar(x1, data_norm1, w, label=labels[0])
        plt.bar(x2, data_norm2, w, label=labels[1])


        # ax.set_xlim(36)
        xlabels = self.motifslabels
        ax.set_xticks(range(1, 37))
        # ax.set_xticks(x)
        ax.set_xticklabels(xlabels)

        plt.xlabel("Patterns")

        ax.set_yticks(pylab.frange(0,1,0.1))
        ax.set_ylim(0, 1)
        plt.ylabel("Value")


        plt.subplots_adjust(left=0.035, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
        plt.legend()
        plt.grid(True)
        fig.savefig(self.simulation_gd_directory + "barplot_"+ self.golden_macrostate_file_indentifier + "_vs_"+self.simulation_macrostate_file_indentifier+".png")
        plt.show()



    def compute(self):

        # ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = self.get_dynamics_golden()
        ALL_TIME_INDEXES = range(0,1)
        golden_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.golden_gd_directory, self.golden_macrostate_file_indentifier)


        simulation_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.simulation_gd_directory, self.simulation_macrostate_file_indentifier)

        error_x = self.compute_error_by_time(golden_temporalmotif_by_time, simulation_temporalmotif_by_time)

        print error_x

        np.savetxt(self.simulation_gd_directory + self.golden_macrostate_file_indentifier + '.error', error_x, delimiter=',', fmt='%f')

        # self.view_multiple_temporalmotif([golden_temporalmotif_by_time, simulation_temporalmotif_by_time], ["Temporal Motif by day(Bitcoin)", "Temporal Motif by day(Simulation)"])

        self.view_multiple_bar([golden_temporalmotif_by_time[0], simulation_temporalmotif_by_time[0]], ["Bitcoin", "Simulation"])


if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()