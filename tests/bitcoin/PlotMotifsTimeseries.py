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

from sklearn.metrics import mean_squared_error
from math import sqrt



class PlotMotifsTimeseries(object):

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


    def plot_bar(self, data, output):
        x = data[0]
        del x[-1]  # removing error heder
        del x[0]  # removing model header now x has just motifs labels
        labels = []
        data_frame = np.zeros((5,len(x)))
        for idx, row in enumerate(data[1:]):
            label = row[0]
            error = row[-1]
            labels.append(label + " e=" + str(round(error,6)))
            y = row
            del y[-1]
            del y[0]
            data_frame[idx] = y


        df = pd.DataFrame(data_frame,columns=x, index=labels)
        ax = df.T.plot.bar(rot=1, figsize=(24,8), grid=True, label='a', zorder=3)
        ax.grid(linestyle='dashed', linewidth=1, alpha=0.4, zorder=0)
        # plt.show()
        plt.savefig(output + '.png')


    def get_motifdata(self, input):
        data = np.genfromtxt(input, dtype=None)
        return data.flatten().tolist()

    def normalize(self, data):
        return (np.array(data) / np.float(np.array(data).max())).tolist() # return data / np.linalg.norm(data)


    def error(self, actual, predicted):
        rms = sqrt(mean_squared_error(actual, predicted))
        return rms



if __name__ == '__main__':
    input_folder = "/Volumes/Ernane/final-data/all-220-days-temporalmotifs/"
    dates_csv= "/Volumes/Ernane/final-data/222-days-label.csv"
    comp = PlotMotifsTimeseries()
    comp.to_csv(input_folder, dates_csv)
