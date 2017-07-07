'''
A library to handle input output operations for the recsys module



Created on July 6, 2017

@author: rafet
'''
import scipy.sparse as sparse
import numpy as np
import gzip
import cjson
import os
import termcolor

import cjson

def writeADictToJson(dictToSave,pathToData):
	'''
    writes a json file and parse it to a python dictionary.
    it is important to note that the whole file is a single json object
    (compared to mongodb structure)
	'''
	fout=open(pathToData,"w")
	fout.write(cjson.encode(dictToSave))
	fout.close()

def readJsonToADict(pathToData):
    '''
    reads a json file and parse it to a python dictionary.
    it is important to note that the whole file is a single json object
    (compared to mongodb structure)
    '''
    fin=open(pathToData,"r")
    JSONDict=cjson.decode(fin.read())
    return JSONDict

    
def redprint(strToPrint):
    termcolor.cprint(strToPrint,"red")
def blueprint(strToPrint):
    termcolor.cprint(strToPrint,"blue")
def greenprint(strToPrint):
    termcolor.cprint(strToPrint,"green")
def colorprint(strToPrint,color="magenta"):
    termcolor.cprint(strToPrint,color)


def does_exist(directory):
    return (True if os.path.exists(directory) else False)
def mkdir_if_not_exist(directory):
    '''
        Create a directory if it does not exist
    '''
    if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        colorprint("[WARNING] folder {0} already exists, doing nothing".format(directory))

def saveTensorJSON(pathToFileName,tensor,verbose=-1,compressed=True):
    '''

    '''
    if(compressed):
        fout=gzip.open(pathToFileName,"wb")
    else:
        fout=open(pathToFileName,"w")

    ithPlayerInd=0
    for x in tensor:
        if(verbose>0 and ithPlayerInd%verbose==0):
            print "saved {0} players".format(ithPlayerInd)
        x_coo=x.tocoo()
        row=(x_coo.row).tolist()
        col=(x_coo.col).tolist()
        data=(x_coo.data).tolist()
        shape=list(x_coo.shape)
        tensorDict={"row":row,"col":col,"data":data,"shape":shape}
        fout.write(cjson.encode(tensorDict)+"\n")
        ithPlayerInd+=1
    fout.close()


def readTensorJSON(pathToFileName,verbose=-1,compressed=True):
    '''

    '''
    tensor=[]
    if(compressed):
        fin=gzip.open(pathToFileName,"rb")
    else:
        fin=open(pathToFileName,"r")
    ithPlayerInd=0
    for line in fin:
        y=cjson.decode(line)
        sparseMatrix=sparse.csr_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
        tensor.append(sparseMatrix)
        ithPlayerInd+=1
    fin.close()
    return tensor
#### Efficient way of saving scipy sparse matrices

def save_list_as_sparse_matrix(filename,x):
    '''
        A very space efficient way to save sparse scipy matrices
        the function converts the given matrix to coo and
        saves the matrix in a compressed format
    '''

    x_coo=sparse.coo_matrix(x)
    row=x_coo.row
    col=x_coo.col
    data=x_coo.data
    shape=x_coo.shape
    np.savez(filename,row=row,col=col,data=data,shape=shape)

def save_sparse_matrix(filename,x):
    '''
        A very space efficient way to save sparse scipy matrices
        the function converts the given matrix to coo and
        saves the matrix in a compressed format
    '''
    x_coo=x.tocoo()
    row=x_coo.row
    col=x_coo.col
    data=x_coo.data
    shape=x_coo.shape
    np.savez(filename,row=row,col=col,data=data,shape=shape)

def load_sparse_matrix(filename):
    '''
        loads the saved matrix as coo matrix
    '''
    y=np.load(filename)
    z=sparse.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z

def load_sparse_matrix_csr(filename):
    '''
        loads the saved matrix as csr matrix
    '''
    y=np.load(filename)
    z=sparse.csr_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z
def load_sparse_matrix_lil(filename):
    '''
        loads the saved matrix as lil matrix
    '''
    y=np.load(filename)
    z=sparse.lil_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z