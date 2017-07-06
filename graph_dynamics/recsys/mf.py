'''

A module for matrix factorization based recommender systems. The data to be modeled here is ideally preference data indicating user's preference to items
We are given a binary preference matrix encoding P preferences of m users to n items(P^[m x n]) and f being number of latent factors.
The algorithm implemented minimizes:
    E(X,Y)=|| P-XY' ||^2
    where X ^[m x f] and Y ^[n x f]
The prediction is then done based on most similar user item pairs.
Given a user u the top-L recommendations can be found as:
    pu=np.dot(Y,X[u]);
    return np.argsort(pu)[::-1][:L]



Created on July 6, 2017

@author: rafet
'''

import baserecsys
import sys
import numpy.linalg as LA
import numpy as np
import os
import recsysIO
class MFRecsys(baserecsys.BaseRecommender):
    def __init__(self):
        baserecsys.BaseRecommender.__init__(self)
        #defines the training types
        self.trainingTypes={"ALS":self.trainALS,"ALS_Repeated":self.trainALSRepeated,"PureSVD":self.trainPureSVD}
        self.ALSREP=5
        pass
    def trainModel(self, traindatamatrix, trainingType="ALS",algorithmConfigs={"numberOfLatentFactors":3,"numberOfMaxIter":10}):
        '''
        trains the model
            self.trainingTypes initialized in the constructor
            algorithmConfigs
        '''
        if (trainingType not in self.trainingTypes):
            print("Training type {0} is not present. Please select one of the followings\n{1}".format(trainingType,self.trainingTypes))
            sys.exit(1)
        else:
            self.trainingTypes[trainingType](traindatamatrix,algorithmConfigs)


    def eigenvectordecomposition(self,G):
        '''
            a numpy based eigenvector decomposition
            the function returns the sorted eigenvectors and eignevalues
            with respect to the eigenvectors
        '''
        eig_values, eig_col_vectors = LA.eigh(G) #eigenvector decomposition for the hermitian matrices!
        sortedinds = np.argsort(eig_values)[::-1]
        eig_values = eig_values[sortedinds]
        eig_col_vectors = eig_col_vectors[:,sortedinds]
        return eig_values, eig_col_vectors


    def svd_partition(self,A):
        '''
            partitions a matrix following singular value decomposition using eigenvalue decomposition.
            for A=USV.T
            where U left singular matrix (column orthonormal)
                  S contains the singular values(positive)
                  V is the right singular matrix (column orthonormal)


        '''

        #first calculate the eigen decomposition
        G=np.dot(A.T,A)
        l,K=self.eigenvectordecomposition(G)
        r=np.where(l>0)[0].shape[0]
        #S contains the positive eigenvalues of the gram matrix
        S=np.diag(np.sqrt(l[:r]))
        #V contains the sorted eigenvector of the gram matrix
        V=K[:,:r]
        #calculate the inverse of the singular matrix to compute the left singular matrix
        Sinv=np.diag(np.sqrt(l[:r])**-1.)
        #left singular matrix
        U=np.dot(A,np.dot(V,Sinv))

        return U,S,V

    def getTruncatedSVDFactors(self,U, S, V,k=3):
        return U[:,:k],S[:k,:k],V[:,:k]

    def trainPureSVD(self,traindatamatrix,algorithmConfigs):
        print("PURE SVD")
        numberOfLatentFactors=algorithmConfigs["numberOfLatentFactors"]


        self.P=traindatamatrix

        U,S,V=self.svd_partition(traindatamatrix)
        Uk,Sk,Vk=self.getTruncatedSVDFactors(U,S,V,numberOfLatentFactors)
        sqrt_of_Sk=np.sqrt(Sk)

        self.X=np.dot(Uk,sqrt_of_Sk)
        self.Y=np.dot(Vk,sqrt_of_Sk)
        error=LA.norm(self.P-np.dot(self.X,self.Y.T))
        print("\t[Training iteration {0}] Error:{1}".format(0,error))
        # self.configs["errorList"].append(error)

    def trainMultipleALS(self,traindatamatrix,algorithmConfigs):
        '''
            runs the ALS algorithm multiple times to get the bests with training error

        '''
        print("Training with ALS algorithm with configs:\n{0}".format(algorithmConfigs))
        numberOfUsers,numberOfItems=traindatamatrix.shape
        numberOfLatentFactors=algorithmConfigs["numberOfLatentFactors"]
        numberOfIterations=algorithmConfigs["numberOfMaxIter"]
        verbose= algorithmConfigs["verbose"] if "verbose" in algorithmConfigs.keys() else -1

        #user preference matrix
        self.P=traindatamatrix
        #user feature matrix
        self.X=self.createRandomMatrix(shapeOfMatrix=(numberOfUsers,numberOfLatentFactors))
        #item feature matrix
        self.Y=self.createRandomMatrix(shapeOfMatrix=(numberOfItems,numberOfLatentFactors))
        self.configs["errorList"]=[]
        error=LA.norm(self.P-np.dot(self.X,self.Y.T))
        for qthIteration in range(numberOfIterations):
            if(verbose!=-1 and qthIteration%verbose==0):
                print("\t[Training iteration {0}] Error:{1}".format(qthIteration,error))

            #update user factors

            self.X=np.dot(self.P,np.dot(self.Y,LA.inv(np.dot(self.Y.T,self.Y))))
            
            self.Y=np.dot(self.P.T,np.dot(self.X,LA.inv(np.dot(self.X.T,self.X))))

            error=LA.norm(self.P-np.dot(self.X,self.Y.T))
            self.configs["errorList"].append(error)
        self.configs["error"]=(error)

    def trainALS(self,traindatamatrix,algorithmConfigs):
        print("Training with ALS algorithm with configs:\n{0}".format(algorithmConfigs))
        numberOfUsers,numberOfItems=traindatamatrix.shape
        numberOfLatentFactors=algorithmConfigs["numberOfLatentFactors"]
        numberOfIterations=algorithmConfigs["numberOfMaxIter"]
        verbose= algorithmConfigs["verbose"] if "verbose" in algorithmConfigs.keys() else -1

        #user preference matrix
        self.P=traindatamatrix
        #user feature matrix
        self.X=self.createRandomMatrix(shapeOfMatrix=(numberOfUsers,numberOfLatentFactors))
        #item feature matrix
        self.Y=self.createRandomMatrix(shapeOfMatrix=(numberOfItems,numberOfLatentFactors))
        self.configs["errorList"]=[]
        error=LA.norm(self.P-np.dot(self.X,self.Y.T))
        for qthIteration in range(numberOfIterations):
            if(verbose!=-1 and qthIteration%verbose==0):
                print("\t[Training iteration {0}] Error:{1}".format(qthIteration,error))

            #update user factors

            self.X=np.dot(self.P,np.dot(self.Y,LA.inv(np.dot(self.Y.T,self.Y))))
            
            self.Y=np.dot(self.P.T,np.dot(self.X,LA.inv(np.dot(self.X.T,self.X))))

            error=LA.norm(self.P-np.dot(self.X,self.Y.T))
            self.configs["errorList"].append(error)
        self.configs["error"]=(error)


    def trainALSRepeated(self,traindatamatrix,algorithmConfigs):
        print("Training with ALS algorithm with configs:\n{0}".format(algorithmConfigs))
        numberOfUsers,numberOfItems=traindatamatrix.shape
        numberOfLatentFactors=algorithmConfigs["numberOfLatentFactors"]
        numberOfIterations=algorithmConfigs["numberOfMaxIter"]
        verbose= algorithmConfigs["verbose"] if "verbose" in algorithmConfigs.keys() else -1

        #user preference matrix
        P=traindatamatrix
        listOfResults=[]
        errorListGlobal=[]
        for ithRun in range(self.ALSREP):
            #user feature matrix
            X=createRandomMatrix(shapeOfMatrix=(numberOfUsers,numberOfLatentFactors))
            #item feature matrix
            Y=createRandomMatrix(shapeOfMatrix=(numberOfItems,numberOfLatentFactors))
            configs["errorList"]=[]
            error=LA.norm(P-np.dot(X,Y.T))
            for qthIteration in range(numberOfIterations):
                if(verbose!=-1 and qthIteration%verbose==0):
                    print("\t[Training iteration {0}] Error:{1}".format(qthIteration,error))

                #update user factors

                X=np.dot(P,np.dot(Y,LA.inv(np.dot(Y.T,Y))))
                
                Y=np.dot(P.T,np.dot(X,LA.inv(np.dot(X.T,X))))

                error=LA.norm(P-np.dot(X,Y.T))
                configs["errorList"].append(error)
            configs["error"]=(error)
            listOfResults.append(X,Y,configs)
            errorListGlobal.append(configs["error"])

        self.X,self.Y,self.configs=listOfResults[np.argmin(errorListGlobal)]

    def getEmbeddingsUnseenData(self,P_unseen,pathToOutputFolder):


        Y_unseen=np.dot(P_unseen.T,np.dot(self.X,LA.inv(np.dot(self.X.T,self.X))))
        error=LA.norm(P_unseen-np.dot(self.X,Y_unseen.T))
        np.savetxt(pathToOutputFolder+"/"+"Y_unseen.csv",Y_unseen,fmt="%s",delimiter=",")
        configs={"error":error,"error_rate":error/LA.norm(P_unseen)}
        recsysIO.writeADictToJson(configs,pathToOutputFolder+"/"+"config_unseen.json")


    def saveTraining(self):

        if(self.mainDirectoryForResults==None):
            recsysIO.redprint("[ERROR] the directory has not been created call prepareANewRun first\nexiting...")
            sys.exit(1)
        else:
            np.savetxt(self.mainDirectoryForResults+"/"+"X.csv",self.X,fmt="%s",delimiter=",")
            np.savetxt(self.mainDirectoryForResults+"/"+"Y.csv",self.Y,fmt="%s",delimiter=",")
            recsysIO.writeADictToJson(self.configs,self.mainDirectoryForResults+"/"+"config.json")

    def loadTrained(self,pathToResultsDir):
        if(recsysIO.does_exist(pathToResultsDir)==False):
            recsysIO.redprint("[ERROR] directory {0} does not exist\nexiting...".format(pathToResultsDir))
        else:
            self.X=np.loadtxt(pathToResultsDir+"/"+"X.csv",delimiter=",")
            self.Y=np.loadtxt(pathToResultsDir+"/"+"Y.csv",delimiter=",")
            self.configs=recsysIO.readJsonToADict(pathToResultsDir+"/"+"config.json")

    def basicPredict(self,mUserIndex,mProductIndex):
        return np.dot(self.Y[mProductIndex],self.X[mUserIndex])
    def basicPredictSingleUser(self,mUserIndex,topL=5):

        #estimated similarities of items to the user:
        p_hat_u=np.dot(self.Y,self.X[mUserIndex])
        topLToPredict=np.argsort(p_hat_u)[::-1][:topL]

        return p_hat_u,topLToPredict




