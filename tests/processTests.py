'''
Created on May 24, 2015

@author: cesar
'''

import json
import unittest
import matplotlib.pyplot as plt
import numpy as np

from bayesian_networks.random_measures import process

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

        
    
if __name__ == '__main__':
    import sys;sys.argv = ['', 'Test.generateBetaProcess']
    unittest.main()