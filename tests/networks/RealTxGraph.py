'''
Created on July 20, 2017

@author: ernaneluis
'''
import operator
import unittest
from itertools import groupby
# import matplotlib.pyplot as plt
import networkx as nx
import pymongo
from pymongo import MongoClient
import datetime
import psycopg2
import psycopg2.extras
from graph_dynamics.communities.bigclam import BigClam
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics import Macrostates

from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.dynamics import FromFilesDynamics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import random

class Test(unittest.TestCase):


    def migrateToSQL(self):
        conn = psycopg2.connect(database="bitcoin", user="postgres", password="postgres", host="0.0.0.0", port="5432")
        cur = conn.cursor()
        print "Postgres open  successfully"


        client       = MongoClient('mongodb://0.0.0.0:27017/')
        db           = client.dump
        collection   = db['blocks12_aggregation']
        print "Mongo open  successfully"

        total = collection.find({}).count()
        print("total" + str(total))

        for vezes in range(100):

            transactions = collection.find({})

            data = []

            page_size = 100000

            for transaction in transactions[:page_size]:
                a_s = 0
                a_r = 0
                tx = None
                r = None
                s = None
                t = 0

                if("a_s" in transaction):
                    a_s = transaction["a_s"]

                if ("a_r" in transaction):
                    a_r = transaction["a_r"]

                if ("tx" in transaction):
                    tx = transaction["tx"]

                if ("r" in transaction):
                    r = transaction["r"]

                if ("s" in transaction):
                    s = transaction["s"]

                if ("t" in transaction):
                    t = transaction["t"]

                time = datetime.fromtimestamp(t)

                tup =  (time, s, r, a_s, a_r, tx )
                data.append(tup)

                result = collection.delete_one({'_id': transaction["_id"]})


            insert_query = 'INSERT INTO transactions (time, sender, receiver, amount_s, amount_r, tx) values %s'
            psycopg2.extras.execute_values(
                cur, insert_query, data, template=None, page_size=page_size
            )

            conn.commit()
            print(str(vezes) + "/100")

        conn.close()

    def get_graph(self):

        graph_series = self.get_graph_series('2017-05-20 16:55:48', '2017-05-24 16:55:48', "day")

        gd_directory = "/Users/ernaneluis/Developer/graph-dynamics/simulations/tx_gd/"

        self.save(graph_series, gd_directory)

        self.visualize(graph_series)

    def get_graph_series(self, date_start, date_end, type):
        # https://www.postgresql.org/docs/8.1/static/functions-formatting.html
        # https://rajivrnair.github.io/postgres-grouping

        types = ["YYYY-MM-DD HH24", "YYYY-MM-DD", "YYYY-MM", "YYYY-Q", "YYYY"]

        if(type == "hour"):
            type_query = types[0]
        elif (type == "day"):
            type_query = types[1]
        elif (type == "month"):
            type_query = types[2]
        elif (type == "quarter"):
            type_query = types[3]
        elif (type == "year"):
            type_query = types[0]

        conn = psycopg2.connect(database="bitcoin", user="postgres", password="postgres", host="0.0.0.0", port="5432")
        cur = conn.cursor()
        print "Postgres open  successfully"

        query = """ select to_char(time, %s) as time_window, time, sender, receiver, amount_s, amount_r     from transactions \
                 WHERE  time BETWEEN %s::timestamp and %s::timestamp \
                 group by time_window, time, sender, receiver, amount_s, amount_r \
                 order by time asc; \
                """
        data = (type_query, date_start, date_end)
        cur.execute(query, data)
        rows = cur.fetchall()


        sorted_input        = sorted(rows, key=operator.itemgetter(0))
        groups              = groupby(sorted_input, key=operator.itemgetter(0))
        group_by_timestamp  = [{'time_step': t, 'items': [x for x in v]} for t, v in groups]

        graphs              = []

        for time_step in group_by_timestamp:
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

        print("TOTAL GRAPHS:" + str(len(graphs)) )
        conn.close()
        return graphs

    def save(self, graph_series, path, name):
        # data = json_graph.node_link_data(G)
        # json.dump(data, open(path, 'w'), indent=2)
        # path/to/folder/{name} _gGD_{id}_.gd
        for idx, G in enumerate(graph_series):
            fname = path + name +"_gGD_" + str(idx) + "_.gd"
            nx.write_edgelist(G, fname, data=True)

    # def load(self, path):
    #     data = json.load(open(fname))
    #     return json_graph.node_link_graph(data)

        # loaded_graph = self.load("/Users/ernaneluis/Developer/graph-dynamics/simulations/tx/4.graph")
        # nx.draw(loaded_graph)
        # plt.show()

    def get_graph_series_mongo(self):

        # d = datetime.date(2017, 1, 1)
        # dtt = d.timetuple()
        # ts = time.mktime(dtt)  # 28800.0


        client = MongoClient('mongodb://0.0.0.0:27017/')
        bitcoinTxDb = client.dump
        transactions = bitcoinTxDb['blocks2_aggregation']

        min_window = 1484711444
        max_window = min_window + 3*3600 # 1 hour

        result = transactions.find({
            "t": {
                '$gte': min_window,
                '$lte': max_window
            }
        })


        graphs = []
        txs = [] #dict of transactions
        for tx in result:
            txs.append(tx)

        sorted_input = sorted(txs, key=operator.itemgetter("t"))
        groups = groupby(sorted_input, key=operator.itemgetter("t"))

        # a = [{'type': time, 'items': [x for x in v]} for time, v in groups]
        group_by_timestamp = [{'time': t, 'items': [x for x in v]} for t, v in groups]
        print(group_by_timestamp)

        for time_step in group_by_timestamp:
            G = nx.Graph()
            print(time_step["time"])
            for idx, transaction in enumerate(time_step["items"]):
                print(transaction)

                s = transaction["s"]
                r = transaction["r"]
                print("tx id " + str(s))
                G.add_node(s)
                G.add_node(r)
                G.add_edge(s, r)

            graphs.append(G)

        print(len(graphs))


    ############### VISUALIZE FUNCTIONS ###############

    def visualize(self, graph_series):
        for idx, graph in enumerate(graph_series):
            nx.draw(graph)
            plt.pause(3)
            plt.clf()
        plt.show()

    def visualize_bigclam(self, gd_directory, totalIndex):
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
        matrices = []

        for i in range(totalIndex):
            dict = MacrostatesHandlers.time_index_macro(gd_directory,
                                                   macro_state_identifier="bigclam",
                                                   macrostate_file_indentifier= "tx_macro",
                                                   time_index=i)

            n       = len(dict)
            m       = len(dict["0"])
            matrix  = np.zeros( (n,m) )

            for idx, key in enumerate(dict):
                matrix[int(key)]=dict[key]

            matrices.append(matrix)


        for id, matrix in enumerate(matrices):
                plt.matshow(matrix)
                fig = plt.gcf()
                plt.clim()  # clamp the color limits
                plt.colorbar()
                plt.pause(1)

        plt.show()

    def visualize_degree(self, gd_directory, totalIndex, macro_state_identifier):

        lists = []

        for i in range(totalIndex):
            dict = MacrostatesHandlers.time_index_macro(gd_directory,
                                                        macro_state_identifier=macro_state_identifier,
                                                        macrostate_file_indentifier="tx_macro",
                                                        time_index=i)
            lists.append(dict)

        data = {}

        for idx, dict in enumerate(lists):
            for idy, key in enumerate(dict):
                if data.has_key(key):
                    data[key].append(dict[key])
                else:
                    data[key] = [dict[key]]

        max_len = max ( [len(data[key]) for id, key in enumerate(data)])

        for id, key in enumerate(data):
            y = data[key]
            #  only show the nodes that appears in all graphs
            if len(y) == max_len:
                y = y + [0] * (totalIndex - len(y)) # fill it with zeros
                x = range(totalIndex)
                plt.bar(x, y)

        plt.xlabel('Time')
        plt.ylabel('Degree')
        plt.show()

    def visualize_basic_stats(self, gd_directory):

        macrostates_run_ideintifier = "tx_macro"
        macro_state_identifier      = "basic_stats"
        macro_keys                  = {"number_of_nodes": "scalar", "number_of_edges": "scalar"}

        df = MacrostatesHandlers.TS_dict_macro(gd_directory, macro_state_identifier, macrostates_run_ideintifier, macro_keys)
        # # print df
        df.plot(kind="bar")
        plt.show()

    def visualize_advanced_stats(self, gd_directory, stats=None):

        macrostates_run_ideintifier = "tx_macro"
        macro_state_identifier      = "advanced_stats"
        macro_keys                  = {}

        if stats == None:
            macro_keys = {"max_degree_nodes": "scalar", "total_triangles": "scalar"}
        else:
            macro_keys[stats] = "scalar"


        df = MacrostatesHandlers.TS_dict_macro(gd_directory, macro_state_identifier, macrostates_run_ideintifier, macro_keys)
        # # print df
        ax = df.plot(kind="bar")
        ax.set_xlabel("Time")
        plt.show()

    def apply_macro(self):

        gd_directory = "/Users/ernaneluis/Developer/graph-dynamics/simulations/tx_gd/"

        #
        ALL_TIME_INDEXES, DYNAMICS_PARAMETERS, macroNumbers = gd_files_handler.gd_folder_stats(gd_directory, True)
        #


        macrostates_run_ideintifier = "tx_macro"

        bigclam_nargs = {
                    "max_number_of_iterations": 100,
                    "error": 0.001,
                    "beta": 0.001
                }


        macrostates_names =  [
                                ("basic_stats", ()),
                                ("advanced_stats", ()),
                                ("degree_centrality", ()),
                                ("degree_nodes", ()),
                                ("bigclam", (bigclam_nargs,))
                             ]
        # compute macros
        Macrostates.evaluate_vanilla_macrostates(gd_directory, macrostates_names,macrostates_run_ideintifier)

        self.visualize_advanced_stats(gd_directory)
        self.visualize_basic_stats(gd_directory)
        self.visualize_degree(gd_directory, 5, "degree_centrality")
        self.visualize_degree(gd_directory, 5, "degree_nodes")

        self.visualize_bigclam(gd_directory, 5)




if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.apply_macro']
    unittest.main()