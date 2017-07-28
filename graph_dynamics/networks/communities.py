'''
Created on Jun 13, 2017

@author: cesar
'''
import sys
sys.path.append("../")
import copy
import random
import numpy as np
import networkx as nx
from scipy.stats import bernoulli
from graph_dynamics.networks.datatypes import Graph

from tag2hierarchy.hierarchy import treeHandlers

def get_full_membership_from_states(graph_paths):
    """
    """
    ALL_STATES = []
    for g in graph_paths:
        ALL_STATES.append(g.get_graph_state()["communities"])
    all_communities = {}
    for c,n  in ALL_STATES[0].iteritems():
        all_communities[c] = set(n)
    
    for state_t in ALL_STATES[1:]:
        for c,n_l in state_t.iteritems():
            for n in n_l:
                all_communities[c].add(n)
    full_membership = {int(c):list(n) for c,n in all_communities.iteritems()}
    return full_membership
    
class CommunityGraph(Graph):
    """
    This graph can be used as a handler in order to
    analyse files, it simply holds a networkx graph
    the state is an empty json
    """
    def __init__(self,identifier_string=None,initial_comunities=None,graph_state=None,networkx_graph=None):
        self.name_string = "CommunityGraph"
        self.type_of_network = 1
        self.networkx_graph = networkx_graph
        if identifier_string == None:
            self.identifier_string = "CommunityGraph"
        else:
            self.identifier_string = identifier_string
        
        if initial_comunities != None:#
            self.graph_state = {}
            self.graph_state["communities"] = initial_comunities
        #initialize with json object
        else:
            self.graph_state = copy.copy(graph_state)
            self.networkx_graph = networkx_graph
            self.communities = graph_state["communities"]
            
        Graph.__init__(self,self.name_string,self.identifier_string,self.graph_state)

    def get_graph_state(self):
        """
        This function should return a json object with all 
        parameters required to initialize such a graph 
        """
        return self.graph_state
            
    def get_networkx(self):
        return self.networkx_graph
          
    def get_adjancency_matrix(self):
        return nx.adjacency_matrix(self.networkx_graph)

    def get_edge_list(self):
        return self.networkx_graph.edge
     
    def get_number_of_edges(self):
        return self.networkx_graph.number_of_edges()
     
    def get_number_of_nodes(self):
        return self.networkx_graph.number_of_nodes()    
    
    
class CommunityTodeschiniCaronGraph(Graph):
    def __init__(self,
                 identifier_string,
                 numberOfCommunities,
                 bK,aK,gammaK,gamma_process):
        
        self.bK = bK
        self.aK = aK
        self.gammaK = gammaK
        self.gamma_process = gamma_process
        name_string = "TodeschiniCaron"
        self.numberOfCommunities = numberOfCommunities
        Graph.__init__(self,name_string,identifier_string)
        self.__generate_group_afilliations()
        self.__generate_adjancency_matrix()
        
    def __generate_group_afilliations(self):
        """
        """
        K = self.gamma_process.K
        W = self.gamma_process.W
        self.AffiliationMatrix = []
        for i in range(K):
            AffiliationVector = []
            for p in range(self.numberOfCommunities):
                theta = (W[i])/(self.gammaK[p]*W[i] + self.bK[p])
                AffiliationVector.append(np.random.gamma(shape=self.aK[p],scale=theta)) 
            self.AffiliationMatrix.append(AffiliationVector)
        self.AffiliationMatrix = np.asarray(self.AffiliationMatrix)
    
    def __generate_adjancency_matrix(self):
        """
        """
        affiliation_product = np.dot(self.AffiliationMatrix,np.transpose(self.AffiliationMatrix))
        correct_diagonal = bernoulli.rvs(1.-np.exp(-affiliation_product.diagonal()))
        self.adjancency_matrix = bernoulli.rvs(1.- np.exp(-2*affiliation_product))
        self.adjancency_matrix[np.diag_indices(self.adjancency_matrix.shape[0])] = correct_diagonal
        self.nxgraph = nx.from_numpy_matrix(self.adjancency_matrix)
        
    def get_networkx(self):
        return self.nxgraph
              
    def get_adjancency_matrix(self):
        return self.adjancency_matrix

    def get_edge_list(self):
        raise None
        
    def get_number_of_edges(self):
        return None
    
    def get_number_of_nodes(self):
        return None
    
    def get_graph_state(self):
        return {"state":"graph state undefined!"}
    
#=====================================================
# Our research models
#=====================================================

class HierarchicalMixedMembership(Graph):
    """
    """
    def __init__(self,numberOfNodes,hierarchy,backgroundProbability,inheritanceProbability,prior_dirichlet):
        """
        Parameters:
        
        numberOfNodes: int
            number of nodes in the graph
            
        hierarchy: tag2hierarchy tree object
            this is the skeleton which defines how we nest the communities
            
        backgroundProbability: float
            probability of the biggest community or root node
            
        inheritanceProbability: float
            the erdos rengy is generated by generating bernoulli variables over probabilities, per edge
            
            we start with a base probability which characterize the whole graph "P"
            for succesive bifurcations of the tree we re multiply the probability with  
            inheritanceProbability (this will diminish the probability) finally we
            use 1.- P and generate bernoulli variables
        
        prior_dirichlet: float
            this will define the proportion of nodes in a given community belongs to the children community
            which is nested in the tree
        """
        #self.numberOfNodesInTree = len(hierarchy.treeHandlers.nodeNames(hierarchy))
        #self.nodesPerLevel = hierarchy.treeHandlers.obtainNodesPerLevel(hierarchy)
        
        self.hierarchy = hierarchy
        self.prior_dirichlet = prior_dirichlet
        self.p_0 = inheritanceProbability
        self.numberOfNodes = numberOfNodes 
        self.backGroundProbability = backgroundProbability
        self.P = np.ones((self.numberOfNodes,self.numberOfNodes))*self.backGroundProbability
        
        self.__generateGraph()
        
    def __generateGraph(self):
        """
        This functions uses a hierarchical structure as defined in tag2hierarchy and 
        creates a graph with a hierarchical structure in a erdos renyi fashion, for each node there is 
        a community density associated with it 
        
        """
        self.hierarchy[0].cargo = {"NodesInCommunity":range(self.numberOfNodes),"SizeOfCommunity":self.numberOfNodes}
        for node in treeHandlers.transverseTree(self.hierarchy):    
            print "Obtaining communities for: ",node.name
            try:
                numberOfChildren = len(node.children)
            except:
                continue
            
            #obtaining list of nodes in parent
            numberOfNodes = node.cargo["SizeOfCommunity"]
            NodesInCommunity = node.cargo["NodesInCommunity"]
            #define how many nodes per children
            #alpha = np.repeat(self.prior_dirichlet,numberOfChildren)
            #proportionOfChildrenCommunities = np.random.dirichlet(alpha=alpha)
            #indexStart = map(int,np.random.dirichlet(alpha=proportionOfChildrenCommunities*numberOfNodes))
            indexStart = map(int,np.random.dirichlet(alpha=np.repeat(self.prior_dirichlet,numberOfChildren))*numberOfNodes)
            #now we divide the nodes
            indexSum = 0
            for j, indexS in enumerate(indexStart[:-1]):
                NodesInChildren = NodesInCommunity[indexSum:indexSum+indexS]
                node.children[j].cargo = {"NodesInCommunity":NodesInChildren,
                                          "SizeOfCommunity":len(NodesInChildren)}
                #
                self.P[np.array(NodesInChildren)[:,np.newaxis],np.array(NodesInChildren)] = \
                self.P[np.array(NodesInChildren)[:,np.newaxis],np.array(NodesInChildren)]*self.p_0
                #(treeHandlers.obtainMyLevel(node.children[j])+1)
                
                indexSum += indexS
            j = len(indexStart)-1
            NodesInChildren = NodesInCommunity[indexSum:]
            node.children[j].cargo = {"NodesInCommunity":NodesInChildren,
                                      "SizeOfCommunity":len(NodesInChildren)}
            self.P[np.array(NodesInChildren)[:,np.newaxis],np.array(NodesInChildren)] = \
            self.P[np.array(NodesInChildren)[:,np.newaxis],np.array(NodesInChildren)]*self.p_0
            
        self.P = 1. - self.P
        self.adjancency_matrix = bernoulli.rvs(self.P) 
        self.nxgraph = nx.from_numpy_matrix(self.adjancency_matrix)
    
    def get_probabilities(self):
        return self.P
    
    def get_networkx(self):
        return self.nxgraph
              
    def get_adjancency_matrix(self):
        return self.adjancency_matrix

    def get_edge_list(self):
        return None
        
    def get_number_of_edges(self):
        return None
    
    def get_number_of_nodes(self):
        return None
    
    def get_graph_state(self):
        return {"state":"graph state undefined!"}
    
#===========================================================================
# OLD FUNCTIONS
#===========================================================================

def barabasiAlbertCommunities(numberOfNodesPerCommunities,numberOfBridgesPerCommunity,barabasiParameter=3):
    """
    This graph simply connects a number of alber barabasi graphs
    
    Parameters:
        numberOfNodesPerCommunities: list of int
        numberOfBridgesPerCommunity: list of int
        barabasiParameter: int
    """
    numberOfNodes = sum(numberOfNodesPerCommunities)
    numberOfCommunities = len(numberOfNodesPerCommunities)
    subGraphs = []
    #generate subgrahps
    for c in numberOfNodesPerCommunities:
        subGraphs.append( nx.barabasi_albert_graph(c,barabasiParameter) )

    #renaming
    for i in range(1,numberOfCommunities):
        myMap = dict(zip(subGraphs[i].nodes(),
                     range(max(subGraphs[i-1].nodes())+1,max(subGraphs[i-1].nodes())+1+len(subGraphs[i].nodes()))))
        subGraphs[i] = nx.relabel_nodes(subGraphs[i],myMap)

    #creates full graph
    fullGraph = nx.Graph()
    for c in range(numberOfCommunities):
        fullGraph.add_edges_from(subGraphs[c].edges())


    #who are the bridges
    bridgesInCommunity = []
    for c in range(numberOfCommunities):
        bridgesInCommunity.append(random.sample(subGraphs[c].nodes(),numberOfBridgesPerCommunity[c]))


    e = np.zeros((numberOfCommunities,numberOfCommunities))
    #reconnect
    for c in range(numberOfCommunities):
        otherComm = set(range(numberOfCommunities)).difference(set([c]))
        try:
            totalExtraNodes = np.concatenate([subGraphs[a].nodes() for a in otherComm])
            bridgesNeighbors = random.sample(totalExtraNodes,numberOfBridgesPerCommunity[c])
            for b in bridgesNeighbors:
                for i in range(numberOfCommunities):
                    if(b in subGraphs[i]):
                        e[c,i]+=1
                        e[i,c]+=1

            for bridgeNode, bridgeNeigbor in zip(bridgesInCommunity[c],bridgesNeighbors):
                fullGraph.add_edge(bridgeNode,bridgeNeigbor)
        except:
            print sys.exc_info
            
    # TO DO: check modularity
    for i  in range(numberOfCommunities):
        e[i,i] = subGraphs[i].number_of_edges()

    e = e/fullGraph.number_of_edges()
    Q = 0
    for i in range(numberOfCommunities):
        Q += e[i,i] - (e[i,:].sum())**2
    
    return (fullGraph,subGraphs,Q,bridgesInCommunity)

graph_class_dictionary = {"CommunityGraph":CommunityGraph}