'''
A base class for the recommender systems module



Created on July 6, 2017

@author: rafet
'''

from abc import ABCMeta
import numpy as np
import os
import errno
import scipy.sparse as sparse
import recsysIO

class BaseRecommender():
    __metaclass__ = ABCMeta

    def __init__(self):
        print("Base Recommender's instance")
        self.configs={}
        self.mainDirectoryForResults=None
        pass

    def prepareANewRun(self,pathToOutputFolder):

        recsysIO.mkdir_if_not_exist(pathToOutputFolder)
        self.mainDirectoryForResults=pathToOutputFolder

    def saveTraining(self):
        pass
    def createRandomMatrix(self,shapeOfMatrix=(4,3)):
        return np.random.rand(*shapeOfMatrix)
    def cosineSimilarityVectorAndMatrix(self,v,A):
        return np.dot(A,v)/(np.linalg.norm(A,axis=1) * np.linalg.norm(v))
    
    def basicPredict(self,mUserIndex,mProductIndex):
        pass
    