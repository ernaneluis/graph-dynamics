'''
Created on Mar 13, 2017

@author: cesar
'''
import copy
import matplotlib
import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt
from scipy.integrate import quadrature
from graph_dynamics.utils import functions
from scipy.stats import poisson, beta, expon
from scipy.stats import gamma as gamma_distribution
from graph_dynamics.random_measures.datatypes import CompletlyRandomMeasures, PoissonMeasure
from graph_dynamics.random_measures.normalized_process import ChineseRestaurantProcess, ExtendedChineseRestaurantProcess

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True 
matplotlib.rcParams['pdf.use14corefonts'] = True

#===============================================================
#  PROCESS 
#===============================================================

class GammaProcess(CompletlyRandomMeasures):
    """
    Completly Random Measure Which is Generated From
    Parameters or a json object ofthe form
    
            measure_state["identifier_string"]
            measure_state["sigma"]
            measure_state["alpha"]
            measure_state["tau"]
            measure_state["measure"]["W"]
            measure_state["measure"]["Theta"]
                measure_state["lambda_measure_state"]
                
    """
    def __init__(self,identifier_string=None,sigma=None,tau=None,alpha=None,K=100,measure_state=None):
        name_string = "GammaProcess"
        
        if measure_state == None:
            CompletlyRandomMeasures.__init__(self,name_string,identifier_string,K)
            
            self.identifier_string = identifier_string
            self.sigma = sigma
            self.alpha = alpha
            self.tau = tau
            self.lambda_measure = PoissonMeasure(self.alpha,identifier_string="LambdaMeasure",K=K)
            self.processDefined = False                 #TO DO: check this        
            self.stickBreakingConstruction(K)

            self.measure_state = {"measure_name":name_string,
                                  "identifier_string":self.identifier_string,
                                  "alpha":self.alpha,
                                  "sigma":self.sigma,
                                  "tau":self.tau,
                                  "measure":{"W":self.W.tolist(),"Theta":self.Theta.tolist()},
                                  "lambda_measure_state":self.lambda_measure.get_measure_state()}
            
        else:
            self.measure_state = measure_state
            self.identifier_string = measure_state["identifier_string"]
            self.sigma = measure_state["sigma"]
            self.alpha = measure_state["alpha"]
            self.tau = measure_state["tau"]
            self.lambda_measure = PoissonMeasure(measure_state=measure_state["lambda_measure_state"])
            self.W = np.asarray(measure_state["measure"]["W"])
            self.Theta = np.asarray(measure_state["measure"]["Theta"]) 
            
            self.processDefined = False                 #TO DO: check this
            
    def jump_measure_intensity(self,w):
        return (1./gamma(1-self.sigma))*(w**(-1.-self.sigma))*np.exp(-self.tau*w)
    
    def lambda_measure_intensity(self,theta):
        return functions.uniform_one(theta)
    
    def normalized_random_measure(self,number_of_arrivals,table_and_costumers=None):
        if table_and_costumers == None:
            costumer_seats, Thetas, C = ChineseRestaurantProcess(numberOfCostumers=number_of_arrivals, lambda_measure = self.lambda_measure)
        else:
            costumer_seats, Thetas, C = ExtendedChineseRestaurantProcess(numberOfCostumers=number_of_arrivals, 
                                                                         lambda_measure=self.lambda_measure,
                                                                         tables_and_costumers=table_and_costumers)
        return (costumer_seats, Thetas, C)
    
    def stickBreakingConstruction(self,K):
        """
        Here we follow the algorithm of 
        
        Gamma Processes, Stick Breaking and Variational Inference
        Anirban Roychowdhury
        Brian Kulis
        
        K: is the truncation parameter and indicates the number of atoms accepted for the algorithm
        """
        Theta = self.lambda_measure.generate_points(K)
        W = []
        
        k = 0 
        roundNumber = 1
        while k < K:
            K1 = poisson.rvs(self.lambda_measure.interval_size)
            for i in range(K1):
                Ek = expon.rvs(self.tau)
                Tk = gamma_distribution.rvs(roundNumber,self.alpha)
                W.append( Ek*np.exp(-Tk) ) 
                k+=1
                if k == K:
                    break
            roundNumber += 1
        self.processDefined = True
        
        self.W = np.asarray(copy.copy(W))
        self.Theta = np.asarray(copy.copy(Theta))
        self.K = K
        return (np.asarray(W),np.asarray(Theta))
                            
    def plotProcess(self,plotName=None,saveTo=None): 
        """
        """
        ymin = np.zeros(len(self.W))
        plt.vlines(self.Theta, ymin, self.W)
        plt.plot(self.Theta,self.W,"ro",markersize=12)
        plt.grid(True)
        plt.show()

    def get_measure_state(self):
        return self.measure_state
    
#====================================================================================
# TO DO: DEFINE BETAPROCESS BELOW AS INHERITANCE
#====================================================================================

class BetaProcess:
    
    def __init__(self,c,Omega,B0,B0parameters,B0maximum):
        """
        c: concentration parameter
        Omega: we assume that the space were the measure B0 is defined is given by [0,Omega] \subsect of R
        B0: base measure for the beta process
        
        """
        self.B0 = B0
        self.B0parameters = B0parameters
        self.B0maximum = B0maximum
        self.Omega = Omega
        self.c = c
        self.gamma = quadrature(self.B0, 0., Omega, self.B0parameters)[0]
        self.processDefined = False
        
    def normalizedB0(self,x):
        """
        normalized version of B0 for the inhomogeneous Poisson Process
        """ 
        return (1./self.gamma)*self.B0(x,*self.B0parameters)
    
    def inhomogeneousPoisson(self):
        """
        generates a set of arrivals from a functional form
        using the thinning process
    
        Parameters:
        T: float
        dT: float
        function: function
        functionParameters
        """
        rateBound = self.B0maximum/self.gamma
        T = self.Omega
        J = poisson.rvs(T * rateBound)
        datesInSeconds = np.random.uniform(0., T, J)
        intensities = self.normalizedB0(datesInSeconds) / rateBound
        r = np.random.uniform(0., 1., J)
        arrivals = np.take(datesInSeconds, np.where(r < intensities)[0])
        return arrivals
    
    def generateBetaProcess(self,N):
        """
        Algorithm from 
        
        Hierarchical Beta Process and the Indian Buffet Process
        Romain Thibaux
        Michael I. Jordan
        """
        P = []
        W = []
        for i in range(1,N):
            lamb = (self.gamma*self.c)/(self.c+i-1.)
            K1 = poisson.rvs(lamb)
            newLocations = []
            while len(newLocations) < K1:
                newLocations.extend(self.inhomogeneousPoisson())
            W.append(newLocations[:K1])   
            P.append(beta.rvs(1,self.c + i - 1.,size=K1))
        self.P = np.concatenate(P)
        self.W = np.concatenate(W)
        self.processDefined = True
        return (P,W)
                
    def plotProcess(self,plotName=None,saveTo=None): 
        """
        """
        ymin = np.zeros(len(self.W))
        plt.vlines(self.W, ymin, self.P)
        plt.plot(self.W,self.P,"ro",markersize=12)
        plt.grid(True)
        plt.show()
     