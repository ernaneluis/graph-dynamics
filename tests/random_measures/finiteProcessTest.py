'''
Created on May 2, 2017

@author: cesar
'''

import sys
sys.path.append("../../")

import json
import unittest
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
from graph_dynamics.random_measures import finite_process
from graph_dynamics.random_measures.datatypes import PoissonMeasure
from graph_dynamics.random_measures.finite_process import FiniteGeneralizedGamma
from graph_dynamics.random_measures.finite_process import FiniteStableBeta

class Test(unittest.TestCase):
    
    def generatePoissonMeasure(self):
        def my_lambda_intensity(t,*parameters):
            return np.sin(t*parameters[0])**2.
        
        
        self.interval_size = 10.
        self.intensity = my_lambda_intensity
        self.intensity_parameters = (2*np.pi,)
        self.upper_bound = 2.
        self.whereToPlot = "./Plots/"
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
        
        PM.generate_points(20)
        PM.PlotProcess(self.plotName, showPlot=True)
    
    def generateFiniteGeneralizedGamma(self):
        self.identifier_string = "FGGP Test"
        self.K = 20
        self.sigma = 0.5
        self.tau = 0.5
        self.alpha = 10.2
        self.lambdaMeasure = PoissonMeasure(self.alpha,identifier_string="LambdaMeasure",K=self.K)
        
        self.FGG = FiniteGeneralizedGamma(self.identifier_string,
                                          self.K,
                                          self.sigma,
                                          self.tau,
                                          self.lambdaMeasure)
        
        self.FGG.PlotProcess(plotName="{0}.pdf", saveTo="./Plots/", showPlot=True)
        

    def generateFiniteStableBeta(self):
        self.identifier_string = "FSBP Test"
        self.K = 20
        self.sigma = 0.5
        self.alpha = 10.2
        self.lambdaMeasure = PoissonMeasure(self.alpha,identifier_string="LambdaMeasure",K=self.K)
        
        self.FSB = FiniteStableBeta(self.identifier_string,
                                          self.K,
                                          self.sigma,
                                          self.lambdaMeasure)
        
        self.FSB.PlotProcess(plotName="{0}.pdf", saveTo="./Plots/", showPlot=True)

if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generatePoissonMeasure','Test.generateFiniteGeneralizedGamma','Test.generateFiniteStableBeta']
    unittest.main()