'''
Created on Jun 1, 2015

@author: cesar
'''

import numpy as np
import matplotlib as plt
import numpy as np
from collections import namedtuple
from abc import ABCMeta, abstractmethod


class ExchangableRandomMeasures(object):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta

    def __init__(self, num_dim):
        self.num_dim = num_dim

    @abstractmethod
    def drift(self,t):
        raise NotImplemented()
    
    @abstractmethod
    def diffusion(self,t):
        raise NotImplemented()
    