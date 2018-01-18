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
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt



class PlotMotifs(object):

    def to_csv(self, input_folder, dates_csv):

        dates = pd.read_csv(dates_csv, header=None)
        columns= range(1,37)
        motifs = []
        motifs_norm_by_row = []
        for idx in range(2, 222):
            file = input_folder + "day_gGD_" + str(idx) + "_.temporalmotifcount"
            motif = self.get_motifdata(file)
            motifs.append(motif)

            aux_norm = self.normalize(motif)
            motifs_norm_by_row.append(aux_norm)

        df_motifs = pd.DataFrame(motifs, index=dates, columns=columns)
        df_motifs.to_csv(input_folder + "plot_motifs_timeseries.csv", encoding='utf-8', index=True, float_format='%.8f')

        df_motifs_norm = pd.DataFrame(motifs_norm_by_row, index=dates, columns=columns)
        df_motifs_norm.to_csv(input_folder + "plot_motifs_timeseries_norm.csv", sep=',', index=True, float_format='%.8f')

        # df_motifs = pd.Series(motifs ,index=dates)
        # df_motifs.plot()

        # df = pd.DataFrame(np.random.randn(1000, 4), index=df_motifs.index, columns=columns)

        # df = df.cumsum()
        # df = df.var()
        # https://seaborn.pydata.org/
        # sns.heatmap(df, cmap='viridis')


        # plt.figure()
        # df_motifs_norm.plot()
        # plt.show()


    def plot_bar(self, datas_files, labels, columns, output, normalize=1):

        columns_all = np.genfromtxt("/Volumes/Ernane/final-data/36-temporalmotif-labels-as-row.csv", dtype=str,
                                    delimiter=",")

        columns_selected = np.array(itemgetter(*columns)(columns_all))

        errors = []
        data_frame = []

        for idx, data_file in enumerate(datas_files):

            if normalize == 1:
                data = self.normalize(self.get_motifdata(data_file))
                data = np.array(itemgetter(*columns)(data))
            elif normalize == 2:
                data = self.get_motifdata(data_file)
                data = np.array(itemgetter(*columns)(data))
                data = self.normalize(data)
            else:
                data = self.get_motifdata(data_file)
                data = np.array(itemgetter(*columns)(data))

            data_frame.append(data)

        if normalize:
            errors.append(self.error(data_frame[0], data_frame[-1]))
            errors.append(self.error(data_frame[1], data_frame[-1]))
            errors.append(self.error(data_frame[2], data_frame[-1]))
            errors.append(self.error(data_frame[3], data_frame[-1]))

            labels[0] = labels[0] + " e=" + str(round(errors[0], 6))
            labels[1] = labels[1] + " e=" + str(round(errors[1], 6))
            labels[2] = labels[2] + " e=" + str(round(errors[2], 6))
            labels[3] = labels[3] + " e=" + str(round(errors[3], 6))

        df = pd.DataFrame(data_frame,columns=columns_selected, index=labels)
        print df
        df.to_csv(output + ".csv", float_format='%.8f', encoding='utf-8', sep=',')
        ax = df.T.plot.bar(rot=1, figsize=(24,8), grid=True, label='a', zorder=3, width=0.9)
        ax.grid(linestyle='dashed', linewidth=1, alpha=0.4, zorder=0)
        # plt.show()
        plt.savefig(output + '.png')


    def get_motifdata(self, input):
        data = np.genfromtxt(input, dtype=int, delimiter=" ")
        # data = pd.read_csv(input, delimiter=" ", dtype=np.int32)
        return data.flatten().tolist()

    def normalize(self, data):
        return (np.array(data) / np.float(np.array(data).max())).tolist() # return data / np.linalg.norm(data)


    def error(self, actual, predicted):
        rms = sqrt(mean_squared_error(actual, predicted))
        return rms



if __name__ == '__main__':
    # input_folder = "/Volumes/Ernane/final-data/all-220-days-temporalmotifs/"
    # dates_csv= "/Volumes/Ernane/final-data/222-days-label.csv"
    comp = PlotMotifs()
    # comp.to_csv(input_folder, dates_csv)

    datas = ["/Volumes/Ernane/final-data/golden/daymodel122_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/final-data/golden/daymodel165_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/final-data/golden/daymodel210_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/final-data/activitydriven/activitydriven1515247743001_gGD_0_.temporalmotifcount",
             "/Volumes/Ernane/simulations/memorynewnode1515017380_gd/memorynewnode1515017380_gGD_0_.gd" ]

    labels = ["daymodel122", "daymodel165", "daymodel210", "activitydriven", "simulation"]

    output = "/Volumes/Ernane/final-data/all_motifs_golden_activity_vs_simulation"

    comp.plot_bar(datas, labels, output, normalize=False)