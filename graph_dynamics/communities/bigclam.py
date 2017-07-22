import numpy as np
import networkx as nx


class BigClam():


    def __init__(self, Graph, maxNumberOfIterations, error, beta):
        self.G      = Graph
        self.numberOfNodes       = self.G.get_number_of_nodes()
        self.numberOfEdges       = self.G.get_number_of_edges()
        self.e                   = (2*self.numberOfEdges)/float((self.numberOfNodes*(self.numberOfNodes-1))) #background edge Probability
        self.threshold           = np.sqrt(-np.log(1-self.e))  # calculating the threshold value using background edge probability as input
        self.nodes               = self.G.get_networkx().nodes()


        self.initial_communities = self.init_communities()
        self.numberOfCommunities = len(self.initial_communities)
        self.F                   = self.init_f()



        # F data model
        # self.F[u] = [Fuc1, Fuc2, Fuc3, ..., Fuc{NumberOfCommunities}]

        # 1.step Learning F
        # iteratively update Fu for each u and stop the iteration if the likelihood does not increase (increase less than 0.001%) after we update Fu for all u
        for it in range(maxNumberOfIterations):
            for u, node in enumerate(self.nodes):
                # u = index of the node
                # oldFu   = self.F[u][:]
                deltaL  = self.gradient(u)
                # calculate new value for Fu
                newFu   = self.F[u] + deltaL * beta
                # remove the negative values
                newFu[np.where(newFu < 0)[0]] = 0
                # update Fu
                self.F[u] = newFu

                #TODO: stop the iteration if the likelihood does not increase (increase less than 0.001%) after we update Fu for all u

                # if(np.isnan(np.sum(newFu))):
                #    break
                # print abs(newFu - oldFu)
                # print "old ",oldFu
                # print "new ",newFu


        # 2. step: Find the communities
        self.community_cluster = self.affiliationToCommunities(self.F)
        print("Number of Clusters: " + str(len(self.community_cluster)))
        #3. step: Color then
        self.values  = self.createNodeColorValuesDifferent(self.community_cluster)

    def init_f(self):
        f = np.zeros((self.numberOfNodes, self.numberOfCommunities))

        for idx, u in enumerate(self.nodes):
            for idy, k in enumerate(self.initial_communities):
                if (u in set(k)):
                    f[idx][idy] = 1
        return f

    def init_communities(self):
        # Initialization.
        # To initialize F , we use locally minimal neighborhoods

        # old method
        # self.F = np.random.random( ( len(self.G.nodes()) , numberOfCommunities) )
        init_communities = []
        for u in self.nodes:
            neighborhood        = nx.ego_graph(self.G.get_networkx(), u, radius=1, center=True, undirected=True)
            neighborhood_nodes  = neighborhood.nodes()
            init_communities.append(neighborhood_nodes)

        # remove duplicated neighborhood communities
        return [x for n, x in enumerate(init_communities) if x not in init_communities[:n]]

    # DeltaL
    def gradient(self, u):
        # see equation between (3) and (4) from the paper
        fv_sum                  = self.F.sum(axis=0)
        fu                      = self.F[u][:]
        numberOfCommunities     = self.F.shape[1]
        fv_sum_neighbors_exp    = np.zeros(numberOfCommunities)
        fv_sum_neighbors        = np.zeros(numberOfCommunities)

        node        = self.nodes[u]
        neighbors   = self.G.get_networkx().neighbors(node)
        for v, node in enumerate(neighbors):
            fv                       = self.F[v][:]
            upper = np.exp(-np.dot(fu, fv))
            lower = (1. - np.exp(-np.dot(fu, fv)))
            # TODO: division by zero, is this ok?
            if(lower == 0.0):
                lower = 0.1
            exp                      = upper / lower
            fv_sum_neighbors_exp    += fv * exp
            fv_sum_neighbors        += fv

        fv_sum_not_neighbors = (fv_sum - fu - fv_sum_neighbors)
        deltaL               = fv_sum_neighbors_exp - fv_sum_not_neighbors
        return deltaL

    def affiliationToCommunities(self, affiliation):
        """
        Determining community affiliations step
        # After we learn F, we still have to determine whether u belongs to community c or not from the value of Fuc.

        """

        affiliationL        = affiliation > self.threshold
        # save a sets of nodes which belongs to community_cluster[c] community
        community_cluster                   = [[] for i in range(self.numberOfCommunities)]
        # community_cluster[c] = [ux, ..., uy]


        for i, Fu in enumerate(affiliationL):
            # get community index where Fuc is higher than threshold
            for c in np.where(Fu)[0]:
                community_cluster[c].append(i)

        # e-community
        # To allow for edges between nodes that do not share any community affiliations, we assume an additional community, called the e-community
        # e is the the background edge probability between a random pair of nodes  works well in practice. For all our experiments we set e = pow(10,-8)

        # store all the nodes which belongs to at least one community
        nodes_with_community = []
        for set_of_belonging_nodes in community_cluster:
            nodes_with_community.extend(set_of_belonging_nodes)

        set_of_all_nodes            = set(self.nodes)
        set_of_nodes_with_community = set(nodes_with_community)
        difference                  = set_of_all_nodes.difference(set_of_nodes_with_community)

        alone_community = list( difference )
        alone_community_indices = [i for i, a in enumerate(alone_community)]
        # add the alone_community to the cluster
        community_cluster.append(alone_community_indices)

        # remove duplicated communities
        return [x for n, x in enumerate(community_cluster) if x not in community_cluster[:n]]
        # return community_cluster

    def createNodeColorValuesDifferent(self, Q):
        """
        Give a different color to each community
        return: list of node with the its community color

        """
        N = len(Q)
        Dv = 1. / N
        thisValue = 0.
        valuePerCommunity = [thisValue + i * Dv for i in range(0, N)]

        #  alone community is always green color
        valuePerCommunity[N-1] = 1.0

        values = []
        for n in self.nodes:
            for idx, q in enumerate(Q):
                if(n in set(q)):
                    values.append(valuePerCommunity[idx])
                    break

        return values


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

