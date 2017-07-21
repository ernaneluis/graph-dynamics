'''
Created on July 20, 2017

@author: ernaneluis
'''
from matplotlib.pyplot import pause
from graph_dynamics.utils.bigclam import BigClam
import unittest
import networkx as nx
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
import datetime
from itertools import groupby
import operator

class Test(unittest.TestCase):

    def realTxGraph(self):




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



        # $gte = min
        # $lte = max

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

        for idx, graph in enumerate(graphs):
            # nx.draw(graph)
            # pause(5)
            # plt.clf()

            print("Calculating BigClam for graph: " + str(idx))

            bigClamObj = BigClam(graph, maxNumberOfIterations=1000, error=0.001, beta=0.001)
            print(bigClamObj.F)
            # nx.draw(graph, cmap=plt.get_cmap('jet'), node_color=bigClamObj.values, with_labels=True)
            # pause(5)
            # plt.clf()

        # plt.show()
        # # query get min db.getCollection('blocks2_aggregation').find({}).sort({t:1}).limit(1).pretty()
        # min = 1484711444
        # # query get max db.getCollection('blocks2_aggregation').find({}).sort({t:1}).limit(-1).pretty()
        # max = 1485999334
        # time_min = datetime.datetime.fromtimestamp(int("1484711444")).strftime('%Y-%m-%d %H:%M:%S')
        # time_max = datetime.datetime.fromtimestamp(int("1484712444")).strftime('%Y-%m-%d %H:%M:%S')
        # print(time_min)
        # print(time_max)





if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.realTxGraph']
    unittest.main()