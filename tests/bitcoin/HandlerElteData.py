'''
Created on July 20, 2017

@author: ernaneluis
'''
import os
import sys
import networkx as nx
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import sys;
import unittest
import bisect
from datetime import datetime
import psycopg2
import psycopg2.extras
import operator
from itertools import groupby

class Test(unittest.TestCase):

    #
    def load_data_to_sql(self, input_file):

        conn = psycopg2.connect(database="ernaneluis", user="postgres", password="postgres", host="localhost", port="5434")
        cur = conn.cursor()
        print "Postgres open  successfully"

        data = []
        count = 0
        total = 129178908
        with  open(input_file) as f:
            for row in f:
                line = row.replace("\n", "").split("\t")

                s = line[1]
                r = line[2]
                a_s = 0
                a_r = 0
                tx = None
                time = datetime.fromtimestamp(float(line[3]))

                tup =  (time, s, r, a_s, a_r, tx )
                data.append(tup)

                if len(data) == 4574 :
                    insert_query = 'INSERT INTO transactions_elte (time, sender, receiver, amount_s, amount_r, tx) values %s'
                    psycopg2.extras.execute_values(
                        cur, insert_query, data, template=None, page_size=4574
                    )
                    conn.commit()
                    data = []
                    count = count + 4574
                    print str(count) + "/" + str(total)
    #


    def get_graph_series(date_start, date_end, type):
    # https://www.postgresql.org/docs/8.1/static/functions-formatting.html
    # https://rajivrnair.github.io/postgres-grouping

        types = ["YYYY-MM-DD HH24", "YYYY-MM-DD", "YYYY-MM", "YYYY-Q", "YYYY"]

        if (type == "hour"):
            type_query = types[0]
        elif (type == "day"):
            type_query = types[1]
        elif (type == "month"):
            type_query = types[2]
        elif (type == "quarter"):
            type_query = types[3]
        elif (type == "year"):
            type_query = types[0]

        conn = psycopg2.connect(database="ernaneluis", user="postgres", password="postgres", host="localhost", port="5434 ")
        cur = conn.cursor()
        print "Postgres open  successfully"

        query = """ select to_char(time, %s) as time_window, time, sender, receiver   from transactions_elte \
                    WHERE  time BETWEEN %s::timestamp and %s::timestamp \
                    group by time_window, time, sender, receiver \
                    order by time asc; \
                   """
        data = (type_query, date_start, date_end)
        cur.execute(query, data)
        rows = cur.fetchall()

        sorted_input = sorted(rows, key=operator.itemgetter(0))
        groups = groupby(sorted_input, key=operator.itemgetter(0))
        group_by_timestamp = [{'time_step': t, 'items': [x for x in v]} for t, v in groups]

        graphs = []

        for time_step in group_by_timestamp:
            # print time_step
            G = nx.Graph()
            for idx, transaction in enumerate(time_step["items"]):
                # time_window, time, sender, receiver, amount_s, amount_r
                # print(transaction)
                s = transaction[2]
                r = transaction[3]
                G.add_node(s)
                G.add_node(r)
                G.add_edge(s, r)

            graphs.append(G)

        print("TOTAL GRAPHS:" + str(len(graphs)))
        conn.close()
        return graphs

    def split_to_temporal_format(self, input_file, dir):
        file_count = 0
        day_count = 0
        one_day_in_seconds = 3600

        name = "bitcoin_temporal_"

        fname = dir + name + str(file_count) + ".temporalmotif"
        file = open(fname, "w")

        pass_first_line = False

        with  open(input_file) as f:
            for row in f:
                line = row.replace("\n", "").split("\t")

                s = line[1]
                r = line[2]
                t = int(line[3])

                if pass_first_line == False:
                    day_count = t
                    pass_first_line = True

                file.write(str(s) + " " + str(r) + " " + str(t) + "\n")

                if t > day_count + one_day_in_seconds:
                    file.close()

                    day_count = t
                    file_count = file_count + 1
                    fname = dir + name + str(file_count) + ".temporalmotif"
                    file = open(fname, "w")

    def convert_graph_to_bigclam_format(self, dir, name):

        input_file = dir + name + ".txt"
        fname = dir + name + "_bigclam.txt"
        output_file = open(fname, "w")

        with  open(input_file) as f:
            for row in f:
                line = row.replace("\n", "").split("\t")

                s = line[0]
                r = line[1]

                output_file.write(str(s) + "\t" + str(r) + "\n")

        output_file.close()


    def split_to_bigclam_format(input_file, dir):
        file_count = 0
        day_count = 0
        one_day_in_seconds = 86400

        fname = dir + "bitcoin_bigclam_" + str(file_count) + ".txt"
        file = open(fname, "w")

        pass_first_line = False

        with  open(input_file) as f:
            for row in f:
                line = row.replace("\n", "").split("\t")

                s = line[1]
                r = line[2]
                t = int(line[3])

                if pass_first_line == False:
                    day_count = t
                    pass_first_line = True

                file.write(str(s) + "\t" + str(r) + "\n")

                if t > day_count + one_day_in_seconds:
                    file.close()

                    day_count = t
                    file_count = file_count + 1
                    fname = dir + "bitcoin_bigclam_" + str(file_count) + ".txt"
                    file = open(fname, "w")

    def save(graph_series, path, name):
        # path/to/folder/{name} _gGD_{id}_.gd
        print "Graphs: " + str(len(graph_series))
        for idx, G in enumerate(graph_series):
            fname = path + name + "_gGD_" + str(idx) + "_.gd"
            nx.write_edgelist(G, fname, data=False)
            print "Saved: " + fname

    def bigclam_format_to_gd(self, bigclam_dir):

        time_indexes = map(int, [filename.split("_")[2].replace(".txt", "") for filename in os.listdir(bigclam_dir) if "_f_matrix" in filename])
        time_indexes = sorted(time_indexes)
        graphs = []

        name = "bitcoin"

        for idx in time_indexes:
            open_name = bigclam_dir  +  "bitcoin_bigclam_" + str(idx) + ".txt"
            write_name = bigclam_dir + name + "_gGD_" + str(idx) + "_.gd"
            fh = open(open_name, 'rb')
            networkx_graph = nx.read_edgelist(fh)
            # graphs.append(networkx_graph)
            nx.write_edgelist(networkx_graph, write_name, data=True)
            fh.close()

        return time_indexes, graphs

    # def convert_gd2temporal(self, gd_directory, time_indexes, series_graph):
    #     # creating temporal graph file input
    #     path = gd_directory + "temporal-graph.txt"
    #     file = open(path, "w")
    #     set_nodes = set()
    #
    #     for idx in time_indexes:
    #         graph = series_graph[idx]
    #         nodes = set(graph.nodes())
    #         set_nodes = set_nodes.union(nodes)
    #
    #     indexes = range(0,len(set_nodes))
    #     all_nodes = dict(zip(set_nodes, indexes))
    #     # all_nodes = list(set_nodes)
    #     for idx in time_indexes:
    #         print "Converting graph #: " + str(idx)
    #         graph = series_graph[idx]
    #
    #         total = len(graph.edges())
    #         for idy, edge in enumerate(graph.edges()):
    #
    #             map_to_index_0 = all_nodes.get(edge[0])
    #             map_to_index_1 = all_nodes.get(edge[1])
    #
    #             file.write(str(map_to_index_0) + " " + str(map_to_index_1) + " " + str(idx + 1) + "\n")
    #
    #             if idy % 100 == 0:
    #                 print str(idy) + "/" + str(total)
    #
    #     file.close()
    #     print "Temporal File: " + path
    #     return path
        #
        # def temporal_motif(self, input, output_dir):
        #     # http://snap.stanford.edu/temporal-motifs/code.html
        #     exe_directory = "/Users/ernaneluis/Developer/graph-dynamics/snap-cpp/examples/temporalmotifs/temporalmotifsmain"
        #
        #     output = output_dir + "temporal-graph-counts.txt"
        #     args1 = "-i:" + input
        #     args2 = "-o:" + output
        #     args3 = "-delta:300"
        #     # calling command of snap in c++
        #     subprocess.call([exe_directory, args1, args2, args3])
        #     return output
        #
        # def view(self, input, output_dir):
        #     print "View: " + input
        #     data = np.genfromtxt(input, dtype=None)
        #     # plt.matshow(data, cmap='Blues', interpolation='nearest')
        #     # plt.colorbar()
        #     # plt.show()
        #
        #     fig, ax = plt.subplots()
        #     min_val = 0
        #     max_val = 6
        #     diff = 1
        #
        #     img = ax.imshow(data, cmap='Blues', interpolation='nearest')
        #
        #     for idx, row in enumerate(data.transpose()):
        #         for idy, value in enumerate(row):
        #             ax.text(idx, idy, value, va='center', ha='center')
        #
        #     # set tick marks for grid
        #     ax.set_xticks(np.arange(min_val - diff / 2, max_val - diff / 2))
        #     ax.set_yticks(np.arange(min_val - diff / 2, max_val - diff / 2))
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     plt.colorbar(img)
        #     image = output_dir + "temporal_motif.png"
        #     plt.savefig(image)
        #     print "Image: " + image
        #     plt.show()

    def compute(self):

        input_file            = "/Users/ernaneluis/Developer/master_thesis/bigclam/old_bitcoin_dataset/txedge.txt"
        # self.load_data_to_sql(path)

        gd_directory = "/Users/ernaneluis/Developer/master_thesis/bigclam/old_bitcoin_dataset/temporal_hour/"
        #
        # graph_series = get_graph_series('2009-01-12 04:30:25', '2013-12-28 22:55:20', "day")
        #
        # save(graph_series, gd_directory, "bitcoin")

        # self.split_to_temporal_format(input_file, gd_directory)

        # self.split_to_bigclam_format(input_file, gd_directory)

        # self.bigclam_format_to_gd(gd_directory)


        name = "lt_graph"
        dir  = "/Users/ernaneluis/Developer/master_thesis/bigclam/bitcoin_graphs/"

        # self.convert_graph_to_bigclam_format(dir,name)

        print "done"

if __name__ == '__main__':
    import sys;sys.argv = ['', 'Test.compute']
    unittest.main()
