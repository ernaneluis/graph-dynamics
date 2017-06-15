'''
Created on Jun 13, 2017

@author: cesar
'''
import numpy as np
import networkx as nx
from scipy.stats import bernoulli
from graph_dynamics.networks.datatypes import Graph
from graph_dynamics.random_measures.process import GammaProcess
from graph_dynamics.random_measures.finite_process import FiniteGeneralizedGamma


class CommunityTodeschiniCaronGraph(Graph):
    """
    """
    def __init__(self,
                 identifier_string,
                 numberOfCommunities,
                 bK,aK,gammaK,gamma_process):
        
        self.bK = bK
        self.aK = aK
        self.gammaK = gammaK
        self.gamma_process = gamma_process
        name_string = "TodeschiniCaron"
        self.numberOfCommunities = numberOfCommunities
        Graph.__init__(self,name_string,identifier_string)
        self.__generate_group_afilliations()
        self.__generate_adjancency_matrix()
        
    def __generate_group_afilliations(self):
        """
        """
        K = self.gamma_process.K
        W = self.gamma_process.W
        self.AffiliationMatrix = []
        for i in range(K):
            AffiliationVector = []
            for p in range(self.numberOfCommunities):
                theta = (W[i])/(self.gammaK[p]*W[i] + self.bK[p])
                AffiliationVector.append(np.random.gamma(shape=self.aK[p],scale=theta)) 
            self.AffiliationMatrix.append(AffiliationVector)
        self.AffiliationMatrix = np.asarray(self.AffiliationMatrix)
    
    def __generate_adjancency_matrix(self):
        """
        """
        affiliation_product = np.dot(self.AffiliationMatrix,np.transpose(self.AffiliationMatrix))
        correct_diagonal = bernoulli.rvs(1.-np.exp(-affiliation_product.diagonal()))
        self.adjancency_matrix = bernoulli.rvs(1.- np.exp(-2*affiliation_product))
        self.adjancency_matrix[np.diag_indices(self.adjancency_matrix.shape[0])] = correct_diagonal
        self.nxgraph = nx.from_numpy_matrix(self.adjancency_matrix)
        
    def get_networkx(self):
        return self.nxgraph
              
    def get_adjancency_matrix(self):
        return self.adjancency_matrix

    def get_edge_list(self):
        raise None
        
    def get_number_of_edges(self):
        return None
    
    def get_number_of_nodes(self):
        return None    