'''
Created on May 24, 2015

@author: cesar
'''

import sys
sys.path.append("../../")

import json
import unittest
import matplotlib.pyplot as plt
import numpy as np

from graph_dynamics.random_measures import process

class Test(unittest.TestCase):
    
    def generateBetaProcess(self):
        def my_omega_measure(t,*parameters):
            return np.sin(t*parameters[0])**2.
        
        self.c = 10.
        self.B0 = my_omega_measure
        self.B0parameters = (2*np.pi,)
        self.Omega = 1.
        self.B0maximum = 1.
        BT = process.BetaProcess(self.c,self.Omega,self.B0,self.B0parameters,self.B0maximum)
        B = BT.generateBetaProcess(100)
        BT.plotProcess()

    def generateGammaProcess(self):
        
        self.identifier_string = "GammaProcessTest"
        self.alpha = 20.
        self.sigma = self.alpha
        self.tau = 1000000000.

        G = process.GammaProcess(self.identifier_string,
                                 self.sigma,
                                 self.tau,
                                 self.alpha,
                                 K=100)
        G.plotProcess()    
    
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateGammaProcess']
    unittest.main()