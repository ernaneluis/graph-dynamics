'''
Created on Apr 17, 2017

@author: cesar
'''
import networkx as nx
import json
import unittest
import matplotlib.pyplot as plt
import numpy as np
from bayesian_networks.random_measures import process
from bayesian_networks.dynamics import PittWalker

class Test(unittest.TestCase):
    
    def generateChinese(self):
        def my_omega_measure(t,*parameters):
            return np.sin(t*parameters[0])**2.
        
        #MEASURE VARIABLES
        self.alpha = 10.
        self.tau = 1.
        self.sigma = 10.
        
        #TO BE MODIFIED WITH POISSON MEASURES
        self.lamb = my_omega_measure
        self.lamb_parameters = (2*np.pi,)
        self.lamb_maximum = 1.
        
        #DYNAMICAL VARIABLES
        self.phi = 2.
        self.rho = 1.
        
        dynamicalNetwork = PittWalker.PallaDynamics(self.sigma,
                                                    self.tau,
                                                    self.alpha,
                                                    self.phi,
                                                    self.rho,
                                                    self.lamb,
                                                    self.lamb_parameters,
                                                    self.lamb_maximum)

        print dynamicalNetwork.CRP(20)    
        
    def generateInitialNetwork(self):
        def my_omega_measure(t,*parameters):
            return np.sin(t*parameters[0])**2.
        
        self.alpha = 20.
        self.tau = 1000000000.
        self.sigma = 20.
        self.phi = 2.
        self.rho = 1.
        self.lamb = my_omega_measure
        self.lamb_parameters = (2*np.pi,)
        self.lamb_maximum = 1.
        dynamicalNetwork = PittWalker.PallaDynamics(self.sigma,
                                                    self.tau,
                                                    self.phi,
                                                    self.rho,
                                                    self.lamb,
                                                    self.lamb_parameters,
                                                    self.lamb_maximum)
        
        network = dynamicalNetwork.generateInitialNetwork(100) 
        nx.draw(network)
        plt.show()  
    
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateInitialNetwork']
    unittest.main()