'''
Created on May 3, 2017

@author: cesar
'''

import json
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_dynamics.random_measures import finite_process
from graph_dynamics.networks.datatypes import FiniteProcessGraphs 
from graph_dynamics.random_measures.datatypes import PoissonMeasure
from graph_dynamics.random_measures.finite_process import FiniteGeneralizedGamma

class Test(unittest.TestCase):
    
    def generateFPGraph(self):
        
        self.measure_identifier_string = "FGGP Test"
        self.K = 20
        self.sigma = 0.01
        self.tau = 0.0001
        self.alpha = 10.2
        self.lambdaMeasure = PoissonMeasure(self.alpha,identifier_string="LambdaMeasure",K=self.K)
        
        self.FGG = FiniteGeneralizedGamma(self.measure_identifier_string,
                                          self.K,
                                          self.sigma,
                                          self.tau,
                                          self.lambdaMeasure)
        
        #self.FGG.PlotProcess(plotName="FFG {0}.pdf", saveTo="../", showPlot=True)
        
        self.graph_identifier = "FPG Test"
        self.FiniteProcessGraph = FiniteProcessGraphs(self.graph_identifier,self.FGG)
        nx.draw_networkx( self.FiniteProcessGraph.get_networkx() )
        plt.show
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateFPGraph']
    unittest.main()