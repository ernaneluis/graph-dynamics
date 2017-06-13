'''
Created on May 3, 2017

@author: cesar
'''

import numpy as np

def uniform_one(t,*parameters):
    """
    """
    try: 
        if type(t.shape) == tuple: 
            return np.ones(len(t))
    except:
        return 1.