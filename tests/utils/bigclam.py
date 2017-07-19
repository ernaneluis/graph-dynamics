'''
Created on July 20, 2017

@author: ernaneluis
'''
from matplotlib.pyplot import pause
from graph_dynamics.utils.bigclam import BigClam
import unittest
import networkx as nx
import matplotlib.pyplot as plt


class Test(unittest.TestCase):

    def bigclam(self):

        ###### Karate Graph Test #####
        karate = nx.social.karate_club_graph()

        nx.draw(karate)
        pause(5)
        plt.clf()

        print("show2")
        bigClamObj = BigClam(karate, maxNumberOfIterations=1000, error=0.001, beta=0.001)
        nx.draw(karate, cmap=plt.get_cmap('jet'), node_color=bigClamObj.values, with_labels=True)

        pause(28)



if __name__ == '__main__':
    import sys;

    sys.argv = ['', 'Test.bigclam']
    unittest.main()