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
from operator import itemgetter
import csv
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['axes.grid'] = True
# matplotlib.rcParams['axes.grid.which'] = 'major'
# matplotlib.rcParams['xtick.minor.visible'] = True

class TemporalmotifAnalysisMultiple(object):

    def __init__(self, golden_gd_directory, golden_macrostate_file_indentifier, simulation_gd_directory, simulation_macrostate_file_indentifier, ALL_TIME_INDEXES):
    # golden standard
    # golden_gd_directory = "/Volumes/Ernane/simulations/daymodel122_gd/"
    # golden_macrostate_file_indentifier = "daymodel122"
    #
    # # simulation_gd_directory  = "/Volumes/Ernane/simulations/nullmodel24h_gd/"
    # # simulation_macrostate_file_indentifier = "nullmodel24h"
    #
    # simulation_gd_directory  = "/Volumes/Ernane/simulations/bitcoinmodel1_gd/"
    # simulation_macrostate_file_indentifier = "bitcoinmodel1"


        self.golden_gd_directory = golden_gd_directory # array
        self.golden_macrostate_file_indentifier = golden_macrostate_file_indentifier # array

        self.simulation_gd_directory = simulation_gd_directory
        self.simulation_macrostate_file_indentifier = simulation_macrostate_file_indentifier

        self.ALL_TIME_INDEXES = ALL_TIME_INDEXES

        self.golden_temporalmotif_by_time = [] # array of temporal motifs by time = array of array
        for idx, golden_gd_dir in enumerate(golden_gd_directory):

            golden_gd_name = golden_macrostate_file_indentifier[idx]
            temporal_by_time_i = self.temporalmotif_by_time(ALL_TIME_INDEXES, golden_gd_dir, golden_gd_name)

            self.golden_temporalmotif_by_time.append(temporal_by_time_i)


        self.simulation_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, simulation_gd_directory,
                                                                  simulation_macrostate_file_indentifier) # array


        self.motifslabels = ["$M_{1,1}$", "$M_{1,2}$", "$M_{1,3}$", "$M_{1,4}$", "$M_{1,5}$", "$M_{1,6}$",
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

    def compute_error_by_time(self):

        errors = []
        for idy, item_golden_temporalmotif in enumerate(self.golden_temporalmotif_by_time):
            golden_i_errors_timeline = []
            name_error = self.golden_macrostate_file_indentifier[idy]

            for idx, golden_temporalmotif in enumerate(item_golden_temporalmotif):

                simulation_temporalmotif = self.simulation_temporalmotif_by_time[idx]

                # normalize it
                golden_temporalmotif_norm       = self.normalize(self.trim_data(golden_temporalmotif))
                simulation_temporalmotif_norm   = self.normalize(self.trim_data(simulation_temporalmotif))

                error_i = self.error1(simulation_temporalmotif_norm, golden_temporalmotif_norm)
                golden_i_errors_timeline.append(error_i)


            np.savetxt(self.simulation_gd_directory + name_error + '.error', golden_i_errors_timeline, delimiter=',', fmt='%f')
            errors.append(golden_i_errors_timeline)


        return errors

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


    def view_multiple_bar(self, datas, labels):

        fig = plt.figure(figsize=(31, 14))
        ax = fig.add_subplot(111)

        w = 0.8
        x = np.array(range(1, 37))
        count = len(datas)

        ind = np.arange(36)

        with open(self.simulation_gd_directory + 'results_all_cycle.csv', 'wb') as csvfile:
            csv_file = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            data_all_cycle_headers = ['Model', 'M12','M15', 'M26', 'M32', 'M33', 'M42', 'M44', 'M53', 'M54', 'M55', 'M56']
            csv_file.writerow(data_all_cycle_headers)

            for idx, data in enumerate(datas):
                data = data[0]
                data_norm = self.normalize(data)
                data_all_relevant = itemgetter(*[1,2,3,4,7,8, 11,12,13, 14, 17, 19, 21,22,23, 26, 27, 28, 29])(data_norm) # 19 motifs

                data_all_cycle = itemgetter(*[1,4, 11, 13,14,19, 21, 26, 27, 28, 29])(data_norm) # 11 cycles



                csv_row = [labels[idx]] + map(lambda x:format(x, '.8f'), list(data_all_cycle))

                csv_file.writerow(csv_row)



                # x = x + w

                # plt.bar(ind, data_norm, w, label=labels[idx])
                # id = idx+1
                a = ind+(w*idx)
                rects = ax.bar(a,data_norm, w, label=labels[idx])


            ax.set_xticks(ind + w/2)

            # ax.set_xlim(-w, len(ind) + w)
            # ax.set_xticks(ind + w)
            # ax.autoscale(tight=True)
            # ax.set_xlim(36)

            # ax.set_xticks(range(1, 37))
            xlabels = self.motifslabels
            ax.set_xticklabels(xlabels)
            plt.xlabel("Patterns")


            ax.set_yticks(pylab.frange(0,1,0.1))
            ax.set_ylim(0, 1)
            plt.ylabel("Value")


            plt.subplots_adjust(left=0.035, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
            plt.legend()
            # plt.grid(True)
            fig.savefig(self.simulation_gd_directory + "barplot_vs_"+self.simulation_macrostate_file_indentifier+".png")
            # plt.show()



    # def compute(self):

        # ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = self.get_dynamics_golden()
        # ALL_TIME_INDEXES = range(0,1)


        # error_x = self.compute_error_by_time(golden_temporalmotif_by_time, simulation_temporalmotif_by_time)

        # print error_x

        # np.savetxt(self.simulation_gd_directory + self.golden_macrostate_file_indentifier + '.error', error_x, delimiter=',', fmt='%f')

        # self.view_multiple_temporalmotif([golden_temporalmotif_by_time, simulation_temporalmotif_by_time], ["Temporal Motif by day(Bitcoin)", "Temporal Motif by day(Simulation)"])

        # self.view_multiple_bar([golden_temporalmotif_by_time[0], simulation_temporalmotif_by_time[0]], ["Bitcoin", "Simulation"])

    def trim_data(self, data, type):
        data_return = []
        # all most relevants 19 motifs: triangles and cycles
        if (type == 'all'):
            # all non-zero motifs
            return itemgetter(*[1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 17, 19, 21, 22, 23, 26, 27, 28, 29])(data)
        elif (type == 'cycle'):
            # data_all_cycle_relevant
            return itemgetter(*[4, 11, 14, 21, 26, 27, 28, 29])(data)
        else:
            # all relevants
            return itemgetter(*[2, 3, 4, 8, 11, 14, 17, 21, 22, 23, 26, 27, 28, 29])(data)


    def results_csv(self, datas, labels, trim_type, headers, name):
        return_rows = []
        with open(self.simulation_gd_directory + name + '.csv', 'wb') as csvfile:
            csv_file = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            csv_file.writerow(headers)
            return_rows.append(headers)

            data_simulation = datas[len(datas)-1][0]
            data_simulation_norm = self.normalize(self.trim_data(data_simulation, trim_type))

            for idx, data in enumerate(datas):
                data = data[0]
                data_norm = self.normalize(self.trim_data(data, trim_type))

                error = self.error1(data_simulation_norm, data_norm)

                prettylist2g = lambda l: '[%s]' % ', '.join("%f" % x for x in l)

                print labels[idx]
                print prettylist2g(data)
                print prettylist2g(data_norm)

                csv_row = [labels[idx]] + map(lambda x: format(x, '.8f'), list(data_norm)) + [error]
                csv_file.writerow(csv_row)
                return_rows.append(csv_row)
        return return_rows


    def plot_bar_plotly(self, data, filename):
        plotly.tools.set_credentials_file(username='ernaneluis', api_key='SS9lJwtw6tqLKvkDNKcO')
        bars = []
        x = data[0]
        del x[-1] # removing error heder
        del x[0] # removing model header now x has just motifs labels

        for idx, row in enumerate(data[1:]):
            label = row[0]
            error = row[-1]
            del row[-1]
            del row[0]

            trace = go.Bar(
                x=x,
                y=row,
                name=label+" :e="+str(round(error,6))
            )
            bars.append(trace)


        layout = go.Layout(
            barmode='group',
            width = 1800,
            height = 840
        )

        fig = go.Figure(data=bars, layout=layout)
        # py.iplot(fig, filename=filename)
        py.image.save_as(fig, filename=self.simulation_gd_directory + filename + '.png')

    def plot_bar(self, data, filename):

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
        plt.savefig(self.simulation_gd_directory + filename + '.png')

    def merge_csv(self, filepaths, outputpath):
        fout = open(outputpath, "a")
        for idx, filepath in enumerate(filepaths):
            f = open(filepath)
            if(idx > 0):
                f.next()  # skip the header
            for line in f:
                fout.write(line)
            f.close()  # not really needed
        fout.close()