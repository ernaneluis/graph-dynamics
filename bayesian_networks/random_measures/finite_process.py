'''
Created on May 2, 2017

@author: cesar
'''
import matplotlib 
import numpy as np
from matplotlib import pyplot as plt
from datatypes import FiniteDimensionalProcess

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


class FiniteGeneralizedGamma(FiniteDimensionalProcess):
    """
    Here we Implement a Finite Process of the Form
    
    $$
    \mu_{K}(\cdot) = \sum^{K}_{k=1} W_k \delta_{\theta_k}
    $$
    
    Which is an approximation for the Generalized Gamma Process 
    
    With Levy measure
    $$ 
    \nu(dw,d\theta) = \rho(dw)\lambda(\theta)
    $$
    
    And the Jump Measure
    $$
    \rho(dw) = \frac{1}{\Gamma(1 - \sigma)}w^{-1-\sigma}\exp{(-\tau w)}dw
    $$$
    
     
    """
    def __init__(self,identifier_string,K,sigma,tau,lambda_measure):
        """
        Constructor
        
        Parameters
        
        identifier_string: string to identify object
        K: number of atoms generated 
        
        """
        self.name_string = "FiniteGeneralizedGammaProcess"
        self.identifier_string = identifier_string
        FiniteDimensionalProcess.__init__(self,self.name_string,identifier_string,K)
        self.sigma = sigma
        self.tau = tau
        self.lambda_measure = lambda_measure
        self.interval_size = lambda_measure.interval_size 
        self.GenerateProcess()
        
    def GenerateProcess(self):
        self.Theta = self.lambda_measure.generate_points(self.K)
        G = np.random.gamma(1-self.sigma,1.,size=self.K)
        T = self.__inverseCumulativeFunction(np.random.uniform(0.,1.,size=self.K))
        self.W = G*T
        self.W_complete = sum(self.W)
        
    def GenerateNormalizedProcess(self):
        return None
    
    #=============================================================
    #
    #=============================================================
    def __inverseCumulativeFunction(self,uniformVariable):
        """
        We evaluate the inverse of:
        
        $$
        \frac{\mathbb{I}_{\{ \tau + (\frac{\alpha}{c})^{\frac{1}{\alpha}} \leq t \leq \tau^{-1} \}} \sigma}{- \tau^{\sigma} + \left(\tau + \left(\frac{\sigma}{c}\right)^{\frac{1}{\sigma}}\right)^{\sigma}} t^{- \sigma - 1}
        $$
        
        Which is:
        
        $$
        \left(\frac{\left(\frac{1}{\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}}\right)^{\sigma}}{\tau^{\sigma} y \left(\frac{1}{\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}}\right)^{\sigma} - y \left(\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}\right)^{\sigma} \left(\frac{1}{\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}}\right)^{\sigma} + 1}\right)^{\frac{1}{\sigma}}
        $$
        """        
        A = self.tau + (self.K*self.sigma)**(1./self.sigma)
        A_inv = 1./A
        B = (self.tau**self.sigma)*uniformVariable*(A_inv**self.sigma) 
        B = B - uniformVariable + 1.
        
        return ((A_inv**self.sigma)/B)**(1/self.sigma)