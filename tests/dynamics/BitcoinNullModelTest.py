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
    golden_gd_directory = "/Volumes/Ernane/day_gd/"
    golden_macrostate_file_indentifier = "day"

    simulation_gd_directory  = "/Volumes/Ernane/simulations/activitydriven_gd/"
    simulation_macrostate_file_indentifier = "activitydriven-macros"

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


        ax.set_xlim(36)
        ax.set_xticks(range(1, 37))
        plt.xlabel("Patterns")

        ax.set_yticks(pylab.frange(0,1,0.1))
        ax.set_ylim(0, 1)
        plt.ylabel("Value")

        path = '/Users/ernaneluis/Developer/master_thesis/temporalmotif_patterns/pattern_1.png'

        for i in range(1,37):
            path = '/Users/ernaneluis/Developer/master_thesis/temporalmotif_patterns/pattern_'+str(i)+'.png'
            self.add_logo(fig, path, scale=0.15, x_frac=0.01445*i+0.007, y_frac=0.05)

        plt.subplots_adjust(left=0.035, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
        plt.legend()
        plt.grid(True)
        fig.savefig(self.simulation_gd_directory + "barplot_simulation_vs_nullmodel.png")
        plt.show()

    def add_logo(self, f, path, x_frac=0.5, y_frac=0.5, scale=1, alpha=1):
        """
        Add an image to the figure (not the axes)
        f: a matplotlib figure instance.
        path: the string path to the image to add to the figure.
        x_frac: the fraction of the x dimension of the figure to set the offset to.
            Must be a float.
        y_frac: the fraction of the y dimension of the figure to set the offset to.
            Must be a float.
        scale: the float scale by which to multiply to the image pixel dimensions.
        alpha: the alpha to set the inserted image to

        Set the figure dpi to the same as screen dpi.

        Use this like:
        f = add_logo(f, 'mah_business_logo.png',x_frac=0.5, y_frac=0.5, scale=0.5, alpha=0.15)
        for setting a watermark. This should put the center of the image in the center of the
        figure at half it's original size.
        """
        assert type(x_frac) == float and type(y_frac) == float, "x_frac and y_frac must be floats."
        im = Image.open(path)
        f.set_dpi(96)
        im.thumbnail((int(im.size[0] * scale), int(im.size[1] * scale)), Image.ANTIALIAS)
        img_x, img_y = im.size[0], im.size[1]
        x_offset = int((f.bbox.xmax * x_frac - img_x / 2))
        y_offset = int((f.bbox.ymax * y_frac - img_y / 2))
        f.figimage(im, xo=x_offset, yo=y_offset, origin='upper', zorder=10, alpha=alpha)
        return f

    def view_patterns_by_time(self, series):

        fig = plt.figure(figsize=(31, 14))
        ax = fig.add_subplot(111)

        leng= len(series)
        x = np.array(range(0, leng))

        series_t = np.transpose(series)
        series_n = self.normalize_series(series_t)

        linestyles = ['_', '-', '--', ':']
        markers = []
        for m in Line2D.markers:
            try:
                if len(m) == 1 and m != ' ':
                    markers.append(m)
            except TypeError:
                pass
        styles = markers + [
            r'$\lambda$',
            r'$\bowtie$',
            r'$\circlearrowleft$',
            r'$\clubsuit$',
            r'$\checkmark$']

        # marker=styles[idx%4]

        # color = cm.rainbow(np.linspace(0, 1, 36))

        color = cm.Vega20(np.linspace(0, 1, 36))
        # colors = ['r', 'b', ...., 'w']

        for idx, y in enumerate(series_n):
            if sum(y) > 0:
                print sum(y)
                if idx % 2 == 0:
                    plt.plot(x, y, label='pattern ' + str(idx+1), color=color[idx], marker=styles[idx%4])
                else:
                    plt.plot(x, y, label='pattern ' + str(idx + 1), color=color[idx])

        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e3)
        # ax.set_aspect(1)
        # ax.set_xlim(leng)
        # ax.set_xticks(range(1, leng+1))
        plt.xlabel("Time")

        ax.set_yticks(pylab.frange(0, 1, 0.1))
        ax.set_ylim(0, 1)

        plt.ylabel("Value")

        path = '/Users/ernaneluis/Developer/master_thesis/temporalmotif_patterns/pattern_1.png'

        for i in range(1, 37):
            path = '/Users/ernaneluis/Developer/master_thesis/temporalmotif_patterns/pattern_' + str(i) + '.png'
            # self.add_logo(fig, path, scale=0.15, x_frac=0.01445 * i + 0.007, y_frac=0.05)



        plt.subplots_adjust(left=0.035, right=0.98, top=0.95, bottom=0.11, wspace=0, hspace=0)
        plt.legend()
        plt.grid(True)
        fig.savefig(self.simulation_gd_directory + "temporalmotif_by_time_bitcoin.png")
        plt.show()


    def compute(self):

        # ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = self.get_dynamics_golden()
        ALL_TIME_INDEXES = range(0,222)
        golden_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.golden_gd_directory, self.golden_macrostate_file_indentifier)
        #

        # simulation_temporalmotif_by_time = self.temporalmotif_by_time(ALL_TIME_INDEXES, self.simulation_gd_directory, self.simulation_macrostate_file_indentifier)

        # error_x = self.compute_error_by_time(golden_temporalmotif_by_time, simulation_temporalmotif_by_time)


        # self.view_temporalmif_by_time(golden_temporalmotif_by_time, "Temporal Motif by day(Bitcoin)")
        # self.view_temporalmif_by_time(simulation_temporalmotif_by_time, "Temporal Motif by day(Simulation)")


        # self.view_multiple_temporalmotif([golden_temporalmotif_by_time, simulation_temporalmotif_by_time], ["Temporal Motif by day(Bitcoin)", "Temporal Motif by day(Simulation)"])

        # self.pca(golden_temporalmotif_by_time)


        # self.view_multiple_bar([golden_temporalmotif_by_time[0], simulation_temporalmotif_by_time[0]], ["Bitcoin", "Simulation"])


        self.view_patterns_by_time(golden_temporalmotif_by_time)
if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.compute']
    unittest.main()