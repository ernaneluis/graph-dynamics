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
        
        Parameters:
        ----------
            identifier_string: string to identify object
            K: number of atoms generated 
            sigma: float
            tau: float
            lambda_measure: Poisson Measure
        """
        self.name_string = "FiniteGeneralizedGammaProcess"
        self.identifier_string = identifier_string
        FiniteDimensionalProcess.__init__(self,self.name_string,identifier_string,K)
        self.sigma = sigma
        self.tau = tau
        self.lambda_measure = lambda_measure
        self.interval_size = lambda_measure.interval_size 
        self.W, self.Theta = self.GenerateMeasure()
        
    def GenerateMeasure(self):
        """
        Here we create the measure i.e. we generate the atoms 
        
        $$
        W = \sum^{K}_{i=1}w_i\delta_{\theta_i}
        $$
        
        Returns
        -------
            (self.W,self.Theta)
            
            self.W: list of floats
                    [w_i]
            self.Theta
                    [\theta_i]        
        """
        self.Theta = self.lambda_measure.generate_points(self.K)
        G = np.random.gamma(1-self.sigma,1.,size=self.K)
        T = self.__inverseCumulativeFunction(np.random.uniform(0.,1.,size=self.K))
        self.W = G*T
        self.W_complete = sum(self.W)
        return self.W, self.Theta
    
    def GenerateNormalizedMeasure(self):
        return None
    
    def __inverseCumulativeFunction(self,uniformVariable):
        """
        We evaluate the inverse of:
        
        $$
        \frac{\mathbb{I}_{\{ \tau + (\frac{\alpha}{c})^{\frac{1}{\alpha}} \leq t \leq \tau^{-1} \}} 
        \sigma}{- \tau^{\sigma} + \left(\tau + \left(\frac{\sigma}{c}\right)^{\frac{1}{\sigma}}\right)^{\sigma}} t^{- \sigma - 1}
        $$
        
        Which is:
        
        $$
        \left(\frac{\left(\frac{1}{\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}}\right)^{\sigma}}{\tau^{\sigma} 
        y \left(\frac{1}{\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}}\right)^{\sigma} - 
        y \left(\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}\right)^{\sigma} 
        \left(\frac{1}{\tau + \left(K \sigma\right)^{\frac{1}{\sigma}}}\right)^{\sigma} + 1}\right)^{\frac{1}{\sigma}}
        $$
        """        
        A = self.tau + (self.K*self.sigma)**(1./self.sigma)
        A_inv = 1./A
        B = (self.tau**self.sigma)*uniformVariable*(A_inv**self.sigma) 
        B = B - uniformVariable + 1.
        return ((A_inv**self.sigma)/B)**(1/self.sigma)

class FiniteStableBeta(FiniteDimensionalProcess):
    """
    Here we Implement a Stable beta Process of the Form
    
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
    \rho(dw) = \frac{\Gamma(1+ c)}{\Gamma(1 - \sigma)\Gamma(c + \sigma)}w^{-1-\sigma} (1-w)^{c+\sigma-1} dw
    $$$
    """
    def __init__(self,identifier_string,K,sigma,lambda_measure):
        """
        Constructor
        
        Parameters
        
        identifier_string: string to identify object
        K: number of atoms generated 
        """
        self.name_string = "FiniteStableBetaProcess"
        self.identifier_string = identifier_string
        FiniteDimensionalProcess.__init__(self,self.name_string,identifier_string,K)
        self.sigma = sigma
        self.lambda_measure = lambda_measure
        self.interval_size = lambda_measure.interval_size 
        self.GenerateMeasure()

    def GenerateMeasure(self):
        """
        """
        G = np.random.gamma(1-self.sigma,1.,size=self.K)
        B = np.random.beta(self.sigma,1.,size=self.K)
        SB = G/B # Here I generate K variables \sim BFRY(sigma)
        Exp = -1/self.sigma
        Fact =(self.sigma*self.K)**Exp
        S = [s * Fact for s in SB] # Here I make them into \sim BFRY(1/K,sigma)
        self.W=[s/(s+1) for s in S] # Here I make them into the J's in Eq 15 of Lee, James and Choi
        self.Theta = self.lambda_measure.generate_points(self.K) 
        self.W_complete = sum(self.W)
