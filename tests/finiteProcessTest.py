'''
Created on May 2, 2017

@author: cesar
'''


import json
import unittest
import numpy as np
import matplotlib.pyplot as plt
from bayesian_networks.random_measures import finite_process
from bayesian_networks.random_measures.datatypes import PoissonMeasure
from bayesian_networks.random_measures.finite_process import FiniteGeneralizedGamma

class Test(unittest.TestCase):
    
    def generatePoissonMeasure(self):
        def my_lambda_intensity(t,*parameters):
            return np.sin(t*parameters[0])**2.
        
        
        self.interval_size = 10.
        self.intensity = my_lambda_intensity
        self.intensity_parameters = (2*np.pi,)
        self.upper_bound = 2.
        self.whereToPlot = "../data/"
        self.plotName = "PoissonMeasure.pdf"  
        
        #==============
        # Non lebesque
        #==============
        PM = PoissonMeasure(interval_size=self.interval_size,
                            identifier_string="PoissonMeasure",
                            K=None,
                            isLebesque=False,
                            name_string="PoissonTest",
                            intensity=self.intensity,
                            intensity_parameters=self.intensity_parameters,
                            upper_bound=self.upper_bound)
        
        #PM.generate_points(20)
        PM.PlotProcess(self.plotName, self.whereToPlot)
        
        #==============
        # Lebesque
        #==============
        #PM = PoissonMeasure(interval_size=self.interval_size,
        #                    identifier_string="UniformPoissonMeasure",
        #                    K=None,
        #                    isLebesque=True,
        #                    name_string="PoissonTest")
        
        #PM.generate_points(20)
        #PM.PlotProcess(self.plotName, self.whereToPlot, showPlot=True)
    
    def generateFiniteGeneralizedGamma(self):
        self.identifier_string = "FGGP Test"
        self.K = 20.
        self.sigma = 0.5
        self.tau = 0.5
        self.alpha = 10.2
        self.lambdaMeasure = PoissonMeasure(self.alpha,identifier_string="LambdaMeasure",K=self.K)
        
        self.FGG = FiniteGeneralizedGamma(self.identifier_string,
                                          self.K,
                                          self.sigma,
                                          self.tau,
                                          self.lambdaMeasure)
        
        self.FGG.PlotProcess(plotName="{0}.pdf", saveTo="../", showPlot=True)
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generatePoissonMeasure','Test.generateFiniteGeneralizedGamma']
    unittest.main()