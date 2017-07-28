'''
Created on Jul 4, 2017

@author: cesar
'''
import snap
import math
import time
import pylab
import random
import numpy as np
import networkx as nx
from scipy import stats
from scipy.stats import pareto, norm, bernoulli
import copy

from graph_dynamics.utils import snap_handlers
from graph_dynamics.networks.tx_graph import TxGraph
from graph_dynamics.networks.datatypes import VanillaGraph
from graph_dynamics.networks.communities import CommunityGraph
from graph_dynamics.dynamics.datatypes import GraphsDynamics

def forestParamTuple(paramDict):
    return (paramDict["BurnExpFireP"],paramDict["StartNNodes"],paramDict["ForwBurnProb"],paramDict["BackBurnProb"],paramDict["DecayProb"],paramDict["Take2AmbasPrb"],paramDict["OrphanPrb"])

def communityGraphForEvolution(initial_graph,community_membership):
    initial_subgraph = initial_graph.subgraph(community_membership)
    mapping = dict(zip(initial_subgraph.nodes(),range(len(initial_subgraph.nodes()))))
    initial_subgraph = nx.relabel.relabel_nodes(initial_subgraph,mapping)
    return initial_subgraph

def aggregateRelabelingForFullCommunityMembership(initial_graph,initial_relabeling,relabeling):
    all_community_nodes = []
    if set(initial_relabeling) == set(relabeling.keys()):
        communities = initial_relabeling.keys()
        full_membership = {c:[] for c in communities}
        for c in communities:
            full_membership[c].extend(initial_relabeling[c].values())
            all_community_nodes.extend(initial_relabeling[c].values())
            for  time_labels in relabeling[c]:
                full_membership[c].extend(time_labels.values())
                all_community_nodes.extend(time_labels.values())
    else:
        print "relabeling not maching in community forest fire"
        raise Exception
    
    noise_community  = list(set(initial_graph.nodes()).difference(set(all_community_nodes)))
    full_membership[0] = noise_community
    return full_membership 

class CommunitiesForestFire(GraphsDynamics):
    """
    This is a wrapper for the snap function Forest Fire
    
    The forest fire is started idependently for each community,
    also a background forest fire is defined for the whole graph, 
    in it, the nodes can connect to any community
    """
    def __init__(self, initial_graph,
                 initial_communities,
                 forest_fire_communities_parameters,
                 timeSeriesOfCommunities,
                 DYNAMICAL_PARAMETERS):
        """
        initial_graph: networkx graph
        
        forest_fire_communities_parameters: a dictionary of dictionaries per community with the following keys
        
                        !!!COMMUNITY ZERO CORRESPONDS TO THE WHOLE GRAPH!!!
                        
                        {0:{BurnExpFireP: bool,
                            StartNNodes: int,
                            ForwBurnProb: double,
                            BackBurnProb: double,
                            DecayProb: double,
                            Take2AmbasPrb: double,
                            OrphanPrb: double},
                        1:{...},...}
        
        timeSeriesOfCommunities: numpy array
            the number of new nodes per time step per community
        """
        self.DYNAMICAL_PARAMETERS = DYNAMICAL_PARAMETERS
        if not len(timeSeriesOfCommunities.keys()) == len(forest_fire_communities_parameters.keys()) == (len(initial_communities.keys())):
            print "Wrong input for communities forest fire "
            raise Exception
        
        self.DYNAMICAL_PARAMETERS["DynamicsClassParameters"]={"CommunitiesForestFire":forest_fire_communities_parameters,
                                                              "TimeSeriesOfComunities":list(timeSeriesOfCommunities)} 


        type_of_dynamics = "snap_shot"
        self.numberOfCommunitiesAndNoise = len(forest_fire_communities_parameters.keys())
        self.forestFireModels = {c:snap.TFfGGen(*forestParamTuple(forest_fire_communities_parameters[c])) for c in range(self.numberOfCommunitiesAndNoise)}
        self.timeSeriesOfCommunities = timeSeriesOfCommunities
        self.initial_communities = initial_communities
        GraphsDynamics.__init__(self, DYNAMICAL_PARAMETERS)

    def generate_graphs_paths(self,initial_graph,T):
        """
        Parameters
        ----------
        
        initial_graph:
        T:
        
        Return
        ------
        graph_series = list of Graph objects
            The len of the list must be T, and the first object should be initial graph
            []
        
        """
        T = T -1
        
        if T != 0:
            initial_communities = initial_graph.communities
            print self.latest_index
            #time series full 
            fullTimeSeriesStack = np.array([self.timeSeriesOfCommunities[c] for c in range(0,self.numberOfCommunitiesAndNoise)])
            fullTimeSeriesStackCum = fullTimeSeriesStack.cumsum(axis=1)
    
            fullTimeSeries = fullTimeSeriesStack.sum(axis=0)
            cumFullTimeSeries = fullTimeSeries.cumsum()
            print cumFullTimeSeries
            #################################################################
            #here we guarantee that the nodes are integers to comply with snaps formats
            initial_graph_nx = initial_graph.get_networkx()
            str_int = dict(zip(initial_graph_nx.nodes(),map(int,initial_graph_nx.nodes())))
            initial_graph_nx = nx.relabel_nodes(initial_graph_nx, str_int)
            
            #here we select the subgraphs according to community relabeling
            snap_graphs = {0:snap_handlers.nx_to_snap(initial_graph_nx)} # full graph
            graph_series = {}
            initial_relabeling = {c:dict(zip(range(len(initial_communities[c])),initial_communities[c])) \
                                  for c in range(1,self.numberOfCommunitiesAndNoise)}
            
            # here we initialize the graph series
            graph_series[0] = [snap_handlers.snap_to_nx(snap_graphs[0])]
            for c in range(1,self.numberOfCommunitiesAndNoise):
                snap_graphs[c] = snap_handlers.nx_to_snap(communityGraphForEvolution(initial_graph_nx,
                                                                                     initial_communities[c])) 
                graph_series[c] = [snap_handlers.snap_to_nx(snap_graphs[c])]
            
            self.latest_index             
            #HERE WE EVOLVE THE COMMUNITIES SEPARATLY
            for c in range(1,self.numberOfCommunitiesAndNoise):
                numberOfNodes = fullTimeSeriesStackCum[c][self.latest_index-1]
                #numberOfNodes = self.timeSeriesOfCommunities[c][0]#????
                for i in range(0,T):
                    number_of_new_nodes = self.timeSeriesOfCommunities[c][self.latest_index+i]
                    numberOfNodes += number_of_new_nodes
                    self.forestFireModels[c].SetGraph(snap_graphs[c])
                    self.forestFireModels[c].AddNodes(int(numberOfNodes), True) #HERE IS THE EVOLUTION <<----------
                    new_networkx_graph = snap_handlers.snap_to_nx(snap_graphs[c])
                    graph_series[c].append(new_networkx_graph)
            
            relabeling = self.communityRelabelingForForestFire(T)
            #HERE WE EVOLVE THE FULL GRAPH
    
            #after checking for consistency, we start the time series loop
            for time in range(0,T):
                number_of_new_nodes = self.timeSeriesOfCommunities[0][self.latest_index+time]
                #community zero is background community
                full_nx_graph  = snap_handlers.snap_to_nx(snap_graphs[0])
                # update the edges from the communities
                for c in range(1,self.numberOfCommunitiesAndNoise):
                    new_community_graph = graph_series[c][time+1]
                    relabeled_graph = nx.relabel_nodes(new_community_graph,relabeling[c][time])
                    relabeled_graph = nx.relabel_nodes(relabeled_graph,initial_relabeling[c])
                    full_nx_graph.add_edges_from(relabeled_graph.edges())
                
                try:    
                    numberOfNodes = cumFullTimeSeries[self.latest_index+time] 
                                
                    new_full_snap_graph = snap_handlers.nx_to_snap(full_nx_graph)
                    #numberOfNodes += number_of_new_nodes
                    self.forestFireModels[0].SetGraph(new_full_snap_graph)
                    self.forestFireModels[0].AddNodes(int(numberOfNodes), True)
                    new_nx_graph = snap_handlers.snap_to_nx(new_full_snap_graph)
                    graph_series[0].append(new_nx_graph)
                except:
                    #TODO: LAST INDEX PROBLEM
                    pass
    
    
            self.full_membership = aggregateRelabelingForFullCommunityMembership(graph_series[0][-1],
                                                                                 initial_relabeling,
                                                                                 relabeling)
            
            local_membership_series = self.full_membership_to_time_membership(graph_series,self.full_membership)
            
            self.relabeling = relabeling
            self.initial_relabeling = initial_relabeling
            GRAPH_SERIES = []
            for graph_nx,local_memberships  in zip(graph_series[0],local_membership_series):
                    GRAPH_SERIES.append(CommunityGraph(self.dynamics_identifier,
                                                       initial_comunities=local_memberships,
                                                       networkx_graph=graph_nx))
        else:
            GRAPH_SERIES = [initial_graph]
                                                               
        return  GRAPH_SERIES  
    
    def set_graph_path(self):
        """
        Empirical Data
        """
        return None
        
    def inference_on_graphs_paths(self):
        """
        Learning/Training
        """
        return None
        
    def get_dynamics_state(self):
        """
        """
        return self.DYNAMICAL_PARAMETERS
 
    def full_membership_to_time_membership(self,graph_series,full_membership):
        """
        generate a local membership per graph in order to initialize the graph states 
        
        graph_series: list of networkx graphs
        full_membership: dictionary with communities memberships
        """
        local_membership_series = []
        for graph in graph_series[0]:
            local_membership = {}
            for community,membership in full_membership.iteritems():
                local_membership[community] = list(set(membership).intersection(set(graph.nodes())))
            local_membership_series.append(local_membership)
        return local_membership_series
    
    #==============================================
    # UTILS FOR THE CLASS
    #==============================================
    def communityRelabelingForForestFire(self,numberOfSteps):
        """
        This function provides the dictionary for relabeling the nodes during the dynamics of the community forest fire
        
        Returns:
            relabeling
        """
        self.latest_index
        fullTimeSeriesStack = np.array([self.timeSeriesOfCommunities[c] for c in range(0,self.numberOfCommunitiesAndNoise)])
        fullTimeSeriesStackCum = fullTimeSeriesStack.cumsum(axis=1)
    
        fullTimeSeries = fullTimeSeriesStack.sum(axis=0)
        cumFullTimeSeries = fullTimeSeries.cumsum()
        
        #if self.latest_index == 0:
        relabeling = {c:[] for c in range(1,self.numberOfCommunitiesAndNoise)}
        #else:
        #    relabeling = {c:[{}] for c in range(1,self.numberOfCommunitiesAndNoise)}
            
        for time in range(self.latest_index,self.latest_index+numberOfSteps):
            lower_border = int(cumFullTimeSeries[time - 1])
            for c in range(1,self.numberOfCommunitiesAndNoise):
                #take new node created in the individual dynamics
                newNodesInCommnity = range(int(fullTimeSeriesStackCum[c][time-1]),int(fullTimeSeriesStackCum[c][time]))
                numberOfNewNodes = int(fullTimeSeriesStack[c][time])

                upper_border = lower_border + numberOfNewNodes
                newNames = range(lower_border, upper_border)

                instantaneous_relabeling = dict(zip(newNodesInCommnity,newNames))
                relabeling[c].append(instantaneous_relabeling)
                
                lower_border += numberOfNewNodes

        return relabeling