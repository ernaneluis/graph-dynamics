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
import seaborn as sns; sns.set()

class Test(unittest.TestCase):
    # golden standard

    def normalize(self, data):
        if sum(data) > 0:
            return (np.array(data) / np.float(np.array(data).max())).tolist() # return data / np.linalg.norm(data)
        else:
            return data
        #TODO: check on temporalmotif paper if they count redudency, because if they do only norm by the max

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

    def normalize_series(self, series):
        norm_series = []
        for idx, data in enumerate(series):
            data_norm = self.normalize(data)
            norm_series.append(data_norm)
        return norm_series

    def view_temporalmif_by_time_heatmap(self, d, labels_path, save_path):

        fig = plt.figure(figsize=(31, 14))
        ax = fig.add_subplot(111)

        # d is 222 rows 36 col  it normalize by row
        # data_norm = self.normalize_series(d)
        # df = pd.DataFrame(data_norm)
        # data = df.transpose()


        df = pd.DataFrame(d)
        # d is 36 rows 222 col
        data_t = df.transpose()
        data_norm = self.normalize_series(data_t.values.tolist())
        data = pd.DataFrame(data_norm)
        # labels
        print data
        labels = pd.read_csv(labels_path)
        x_labels = labels.values.flatten().tolist()
        xlabels =[]
        for idx, l in enumerate(x_labels):
            if idx % 2 == 0:
                xlabels.append(l)
            else:
                xlabels.append("")
        xlabels[-1]=x_labels[-1]

        # y label
        ylabels = ["$M_{1,1}$", "$M_{1,2}$","$M_{1,3}$","$M_{1,4}$","$M_{1,5}$","$M_{1,6}$",
             "$M_{2,1}$","$M_{2,2}$","$M_{2,3}$","$M_{2,4}$","$M_{2,5}$","$M_{2,6}$",
             "$M_{3,1}$","$M_{3,2}$","$M_{3,3}$","$M_{3,4}$","$M_{3,5}$","$M_{3,6}$",
             "$M_{4,1}$","$M_{4,2}$","$M_{4,3}$","$M_{4,4}$","$M_{4,5}$","$M_{4,6}$",
             "$M_{5,1}$","$M_{5,2}$","$M_{5,3}$","$M_{5,4}$","$M_{5,5}$","$M_{5,6}$",
             "$M_{6,1}$","$M_{6,2}$","$M_{6,3}$","$M_{6,4}$","$M_{6,5}$","$M_{6,6}$"]

        ax = sns.heatmap(data,  cmap="tab20c_r", xticklabels=xlabels, yticklabels=ylabels, cbar=True, cbar_kws={"orientation": "horizontal", "shrink":0.6, "fraction": 0.10}, linewidths=0.01)

        ax.tick_params(axis='x', which='major', pad=25)
        plt.xticks(fontsize=8   )

        plt.xlabel("Day")
        plt.ylabel("2/3-nodes, 3-edges $\delta$-temporal Motifs")
        plt.subplots_adjust(left=0.045, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
        fig.savefig(save_path)
        plt.show()


    def view_temporalmotifs_by_time(self, d, labels_path, save_path, selected=None):

        fig = plt.figure(figsize=(31, 14))
        ax = fig.add_subplot(111)

        # d is 222 rows 36 col  it normalize by row
        # data_norm = self.normalize_series(d)
        # df = pd.DataFrame(data_norm)
        # data = df.transpose()


        df = pd.DataFrame(d)
        # d is 36 rows 222 col
        data_t = df.transpose()
        data_norm = self.normalize_series(data_t.values.tolist())
        data = pd.DataFrame(data_norm)
        # print data

        # labels

        labels = pd.read_csv(labels_path)
        x_labels = labels.values.flatten().tolist()
        xlabels =[]
        for idx, l in enumerate(x_labels):
            if idx % 2 == 0:
                xlabels.append(l)
            else:
                xlabels.append("")
        xlabels[-1]=x_labels[-1]

        # y label
        ylabels = ["$M_{1,1}$", "$M_{1,2}$","$M_{1,3}$","$M_{1,4}$","$M_{1,5}$","$M_{1,6}$",
             "$M_{2,1}$","$M_{2,2}$","$M_{2,3}$","$M_{2,4}$","$M_{2,5}$","$M_{2,6}$",
             "$M_{3,1}$","$M_{3,2}$","$M_{3,3}$","$M_{3,4}$","$M_{3,5}$","$M_{3,6}$",
             "$M_{4,1}$","$M_{4,2}$","$M_{4,3}$","$M_{4,4}$","$M_{4,5}$","$M_{4,6}$",
             "$M_{5,1}$","$M_{5,2}$","$M_{5,3}$","$M_{5,4}$","$M_{5,5}$","$M_{5,6}$",
             "$M_{6,1}$","$M_{6,2}$","$M_{6,3}$","$M_{6,4}$","$M_{6,5}$","$M_{6,6}$"]

        color = cm.Vega20(np.linspace(0, 1, 36))

        markers = ['_', 'd', '^', 'X', 'o', '*', '+']

        # [[u'D', u's', u'|', u'P', u'x', u'X', u'_', u'^', u'd', u'h', u, u, u',', u, u'.', u'1', u'p', u'3', u'2', u'4', u'H', u'v', u'8', u'<', u'>']

        x = range(0,222)

        dv = data.values
        for idx, y in enumerate(dv):
            print idx
            if selected is not None:
                if idx in selected:
                    if sum(y) > 0:
                        if idx % 2 == 1:
                            ax.plot(x, y, label=ylabels[idx], color=color[idx], marker=markers[idx % len(markers)])
                        else:
                            ax.plot(x, y, label=ylabels[idx], color=color[idx])
            else:
                if sum(y) > 0:
                    if idx % 2 == 1:
                        ax.plot(x, y, label=ylabels[idx], color=color[idx], marker=markers[idx % len(markers)])
                    else:
                        ax.plot(x, y, label=ylabels[idx], color=color[idx])




        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation='vertical')
        ax.tick_params(axis='x', which='major', pad=25)
        plt.xticks(fontsize=7)
        plt.xlabel("Day")

        ax.set_yscale("log")
        plt.ylabel("2/3-nodes, 3-edges $\delta$-temporal Motifs")

        plt.legend()
        plt.subplots_adjust(left=0.045, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
        fig.savefig(save_path)
        plt.show()


    def view_variance_by_pattern(self, d, save_path):

        fig = plt.figure(figsize=(31, 14))
        ax = fig.add_subplot(111)


        df = pd.DataFrame(d)
        # d is 36 rows 222 col
        data_t = df.transpose()
        data_norm = self.normalize_series(data_t.values.tolist())
        data = pd.DataFrame(data_norm)
        # print data

        # labels

        xlabels = ["$M_{1,1}$", "$M_{1,2}$","$M_{1,3}$","$M_{1,4}$","$M_{1,5}$","$M_{1,6}$",
             "$M_{2,1}$","$M_{2,2}$","$M_{2,3}$","$M_{2,4}$","$M_{2,5}$","$M_{2,6}$",
             "$M_{3,1}$","$M_{3,2}$","$M_{3,3}$","$M_{3,4}$","$M_{3,5}$","$M_{3,6}$",
             "$M_{4,1}$","$M_{4,2}$","$M_{4,3}$","$M_{4,4}$","$M_{4,5}$","$M_{4,6}$",
             "$M_{5,1}$","$M_{5,2}$","$M_{5,3}$","$M_{5,4}$","$M_{5,5}$","$M_{5,6}$",
             "$M_{6,1}$","$M_{6,2}$","$M_{6,3}$","$M_{6,4}$","$M_{6,5}$","$M_{6,6}$"]


        dv = data.values
        x = np.array(range(0,len(dv)))

        variances = []
        for idx, y in enumerate(dv):
            variance = np.var(np.array(y))
            variances.append(variance)


        sns.barplot(x, variances, palette="tab20")


        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        plt.xticks(fontsize=10  )
        plt.xlabel("2/3-nodes, 3-edges $\delta$-temporal Motifs" )

        plt.ylabel("Variance of all days")

        plt.subplots_adjust(left=0.045, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
        fig.savefig(save_path)
        plt.show()


    def compute(self):

        golden_gd_directory = "/Volumes/Ernane/day_gd/"
        golden_macrostate_file_indentifier = "day"

        # ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = self.get_dynamics_golden()
        ALL_TIME_INDEXES = range(0,222)
        golden_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, golden_gd_directory, golden_macrostate_file_indentifier)


        # self.view_temporalmif_by_time_heatmap(golden_temporalmotif_by_time, "/Volumes/Ernane/222-days-label.csv", "/Volumes/Ernane/normbyrow_temporalmotif_by_time_heatmap.png")
        #

        # self.view_temporalmotifs_by_time(golden_temporalmotif_by_time, "/Volumes/Ernane/222-days-label.csv",
        #                                  "/Volumes/Ernane/temporalmotif_by_time_series_all.png")


        self.view_temporalmotifs_by_time(golden_temporalmotif_by_time, "/Volumes/Ernane/222-days-label.csv",
                                      "/Volumes/Ernane/1h_temporalmotif_by_time_series_selected.png", selected=[2,3,8,17,23])

        # self.view_variance_by_pattern(golden_temporalmotif_by_time, "/Volumes/Ernane/1h_temporalmotif_by_time_variance.png")
if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()