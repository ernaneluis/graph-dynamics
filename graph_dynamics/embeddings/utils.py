'''
Created on Jun 30, 2017

@author: cesar
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def clusterEmbeddings(G,embeddingsFile,number_of_clusters=2):
    """
    embeddingsFile: string path to embeddings
            
    """
    data = np.loadtxt(embeddingsFile, skiprows=1)
    vecs = data[:, 1:]
    nodes = data[:, 0].astype(np.int)
    c = KMeans(n_clusters=number_of_clusters, random_state=0).fit(vecs).labels_

    X_tsne = TSNE(learning_rate=100).fit_transform(vecs)
    X_pca = PCA().fit_transform(vecs)
    

    #pos = nx.spring_layout(G)
    nx.draw_networkx(G, nodelist=nodes.astype(np.int).tolist(), node_color=c)
    plt.show()
    