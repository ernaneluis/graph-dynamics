import numpy as np
import networkx as nx

import numpy as np
import pickle
import json
from random import random
class BigClam():

    """
    Implementation of the bigClAM algorithm.
    Throughout the code, we will use tho following variables
      * F refers to the membership preference matrix. It's in [NUM_PERSONS, NUM_COMMUNITIES]
       so index (p,c) indicates the preference of person p for community c.
      * A refers to the adjency matrix, also named friend matrix or edge set. It's in [NUM_PERSONS, NUM_PERSONS]
        so index (i,j) indicates is 1 when person i and person j are friends.
    """

    def __init__(self, Graph, numberOfCommunity, maxNumberOfIterations):

        # ////////
        ##generate data
        # number of nodes,
        #  w: assignment probability of a community on the first step,
        # p: p_i is the probability of an edge when two people share community i
        # cross: portion of people to assign a second cluster
        # datagen = Datagen(self.numberOfNodes, [1], [1], .1).gen_assignments().gen_adjacency()
        # p2c = datagen.person2comm
        # adj2 = datagen.adj
        adj = nx.to_numpy_matrix(Graph.get_networkx())

        # F data model
        # self.F[u] = [Fuc1, Fuc2, Fuc3, ..., Fuc{NumberOfCommunities}]

        # 1.step Learning F
        # iteratively update Fu for each u and stop the iteration if the likelihood does not increase (increase less than 0.001%) after we update Fu for all u
        self.F = self.train(adj, numberOfCommunity ,maxNumberOfIterations)
        self.F_argmax = np.argmax(self.F, 1)

        # print self.F
        # data = gen_json(adj, p2c, F_argmax)

        # # with open('../data/data.json','w') as f:
        # with open('ui/assets/data.json', 'w') as f:
        #     json.dump(data, f, indent=4)
        #
        # for i, row in enumerate(F):
        #     print(row)
        #     print(p2c[i])


        # 2. step: Find the communities

    def sigm(self, x):
        return np.divide(np.exp(-1. * x), 1. - np.exp(-1. * x))

    def log_likelihood(self, F, A):
        """implements equation 2 of
        https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
        A_soft = F.dot(F.T)

        # Next two lines are multiplied with the adjacency matrix, A
        # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
        FIRST_PART  = A * np.log(1. - np.exp(-1. * A_soft))
        sum_edges   = np.sum(FIRST_PART)
        SECOND_PART = (1 - A) * A_soft
        sum_nedges  = np.sum(SECOND_PART)

        log_likeli = sum_edges - sum_nedges
        return log_likeli

    def gradient(self, F, A, i):
        """Implements equation 3 of
        https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf

          * i indicates the row under consideration

        The many forloops in this function can be optimized, but for
        educational purposes we write them out clearly
        """
        N, C = F.shape

        neighbours = np.where(A[i])
        nneighbours = np.where(1 - A[i])

        sum_neigh = np.zeros((C,))
        for nb in neighbours[0]:
            dotproduct = F[nb].dot(F[i])
            sum_neigh += F[nb] * self.sigm(dotproduct)

        sum_nneigh = np.zeros((C,))
        # Speed up this computation using eq.4
        for nnb in nneighbours[0]:
            sum_nneigh += F[nnb]

        grad = sum_neigh - sum_nneigh
        return grad

    def train(self, A, C, iterations=100):
        # initialize an F
        N = A.shape[0]
        F = np.random.rand(N, C)
        # print "N nodes: " + str(N) + " times: " + str(iterations*N)
        for n in range(iterations):
            for person in range(N):
                grad = self.gradient(F, A, person)
                F[person] += 0.005 * grad
                F[person] = np.maximum(0.001, F[person])  # F should be nonnegative

            # ll = self.log_likelihood(F, A)
            print("N nodes: " + str(N) + ' step %5i/%5i' % (n, iterations))
        return F
# 1. ingredient:
#       communities arise due to shared group affiliations. we links nodes of the social network to communities that they belong to
#
# 2. ingredient:
#       people tend to be involved in communities to various degrees.
#       The higher the nodes weight of the affiliation to the community the more likely is the node to be connected to other members in the community.
#
# 3. ingredient:
#       when people share multiple community affiliations (e.g., co-workers who attended the same university),
#       the links between them caused by for one dominant reason (i.e., shared community).
#       This means that for each community that a pair of nodes shares,
#       we get an independent chance of connecting the nodes.
#       Thus, naturally, the more communities a pair of nodes shares, the higher the probability of being connected.
#
#       in other words,
#       The observation that the probability of an edge increases as a function of the number of shared communities
#       means that nodes in the overlap of two (or more) communities are more likely to be connected


# Let F be a nonnegative matrix where Fuc is a weight between node u  and community c.
# Fuc is a weight between node u   and community c, where 0 means no affiliation
# Given F, the BIGCLAM generates a graph G(V, E) by creating edge (u, v) between a pair of nodes u, v with probability
# p(u, v) = 1 - exp( dot_product(-Fu, Transpose(Fv)) )
# where Fu is a weight vector for node u
# each community c, connects independently its member nodes u and v, depending on the value of F .




#
# Choosing the number of communities.
# To find the number of communities K, we adopt the approach used in [2].
# We reserve 20% of node pairs as a hold out set. Varying K,
# we fit the BIGCLAM model with K communities on the 80% of node pairs and then evaluate the likelihood of BIGCLAM on the hold out set.
# The K with the maximum hold out likelihood will be chosen as the number of communities.\



# ou r gols is the gerneate a model of agents that creates an behaviour of addres which reproduces an epiracal data



class Datagen():
    def __init__(self, N, w, p, cross):
        """
        Data generator.
        The social network will have N people. At first, we assign each
        person a community with probability w_i. Secondly. we assign a second
        community to a portion (cross) of the community
        :param N: the total number of nodes in the graph
        :param w: assignment probability of a community on the first step
        :param p: p_i is the probability of an edge when two people share community i
        :param cross: portion of people to assign a second cluster
        """
        self.N = N
        self.w = w
        self.p = p
        self.cross = cross

        self.num_comm = len(w)
        assert self.num_comm == len(p)

    @property
    def adj(self):
        return self.A + self.A.T

    def gen_assignments(self):
        # First step
        W = len(self.w)
        initial_comm = np.random.choice(W, p=self.w, size=(self.N,))

        # Second
        person2comm = []
        for n, comm in enumerate(initial_comm):
            all_comm = {comm}
            if random() < self.cross:
                all_comm.add(np.random.randint(0, W))
            person2comm.append(all_comm)

        self.person2comm = person2comm
        return self

    def gen_adjacency(self):
        # TODO check if it is upper traingular
        A = np.zeros((self.N, self.N), dtype=np.int8)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                same_communities = self.person2comm[i].intersection(self.person2comm[j])
                p_nedge = 1.0
                at_least_one = False
                for comm_shared in same_communities:
                    at_least_one = True
                    p_nedge *= 1. - self.p[comm_shared]
                if not at_least_one:
                    p_nedge = 0.99

                p_edge = 1 - p_nedge
                if random() < p_edge:
                    A[i, j] = 1
        self.A = A
        return self
