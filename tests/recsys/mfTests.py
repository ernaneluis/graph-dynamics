'''

A module to test the matrix factorization module


Created on July 6, 2017

@author: rafet
'''

import os
import numpy as np
import sys
sys.path.append("../")


from graph_dynamics.recsys import recsysIO
from graph_dynamics.recsys import mf

import unittest
class Test(unittest.TestCase):
    
    
    def trainALS(self):

        #create a dummy dataset
        # Harry Potter, Avatar, LOTR 3, Gladiator, Titanic, and Glitter
        traindat=np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])

        pathToResults="/tmp/recsys_test_folder/"
        #considering a unix system
        os.system("rm -rf {0}".format(pathToResults))
        recommenderModule=mf.MFRecsys()

        trainingType="ALS"
        algorithmConfigs={"numberOfLatentFactors":2,"numberOfMaxIter":10,"verbose":1}
        recommenderModule.prepareANewRun(pathToResults)
        recommenderModule.trainModel(traindat,trainingType=trainingType,algorithmConfigs=algorithmConfigs)
        recommenderModule.saveTraining()

        #test loading
        recommenderModule2=mf.MFRecsys()
        recommenderModule2.loadTrained(pathToResults)
        playerId=1
        particularItem=1
        estimationForParticularId=recommenderModule2.basicPredict(playerId,particularItem)
        recsysIO.colorprint("estimated value of user {0} for item {1} is {2}".format(playerId,particularItem,estimationForParticularId),"yellow")
        p_hat_u,topLToPredict=recommenderModule2.basicPredictSingleUser(playerId,5)
        p=traindat[playerId]
        recsysIO.colorprint("original matrix of a user is:\n\t{0}".format(p),"cyan")
        recsysIO.colorprint( "prediction for a user is:\n\t{0}\n\t{1}".format(p_hat_u,topLToPredict),"green")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.trainALS']
    unittest.main()


