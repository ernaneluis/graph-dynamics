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
    def generate_graphs_series(self,number_of_steps,output_type):
        """
        Applies the kernel function on every pair of data points between :param x and :param x1.

        In case when :param x1 is None the kernel is applied on every pair of points in :param x.
        :param x: first set of points
        :param x1: second set of points
        :return: distance between every two points
        """
        raise NotImplemented()

    @abstractmethod
    def define_graphs_series(self,graphs_paths,output_type,dynamical_process=None):
        """
        Applies the kernel function on every pair of data points between :param x and :param x1.

        In case when :param x1 is None the kernel is applied on every pair of points in :param x.
        :param x: first set of points
        :param x1: second set of points
        :return: distance between every two points
        """
        raise NotImplemented()    
    
    #===============================================
    # For everyone
    #===============================================
    def macrostates_series(self,T,macrostate):
        """
        Applies the kernel function on every pair of data points between :param x and :param x1.

        In case when :param x1 is None the kernel is applied on every pair of points in :param x.
        :param x: first set of points
        :param x1: second set of points
        :return: distance between every two points
        """
        return None