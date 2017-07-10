'''
Created on Jun 9, 2017

@author: cesar
'''

from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt
import numpy as np


class GraphsDynamics(object):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta

    def __init__(self, initial_network,type_of_dynamics,dynamics_parameters):
        self.initial_network = initial_network 
        self.type_of_dynamics = type_of_dynamics

    @abstractmethod
    def generate_graphs_paths(self,number_of_steps,output_type,keep_path_in_memory=True):
        """
        """
        raise NotImplemented()

    @abstractmethod
    def inference_on_graphs_paths(self,graphs_paths,output_type,dynamical_process=None):
        """
        """
        raise NotImplemented()    
    
    #===============================================
    # For everyone
    #===============================================
    def macrostates_series(self,T,macrostate,macrostate_string):
        """
        """
        return None