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

from graph_dynamics.dynamics.datatypes import GraphsDynamics
from graph_dynamics.networks.tx_graph import TxGraph
from graph_dynamics.utils import snap_handlers

    
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
    def __init__(self, initial_graph,initial_communities,forest_fire_communities_parameters,timeSeriesOfCommunities):
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
        if not len(timeSeriesOfCommunities.keys()) == len(forest_fire_communities_parameters.keys()) == (len(initial_communities.keys())):
            print "Wrong input for communities forest fire "
            raise Exception

        type_of_dynamics = "snap_shot"
        GraphsDynamics.__init__(self, initial_graph, type_of_dynamics, None)
        self.initial_graph = initial_graph
        self.numberOfCommunitiesAndNoise = len(forest_fire_communities_parameters.keys())
        self.forestFireModels = {c:snap.TFfGGen(*forestParamTuple(forest_fire_communities_parameters[c])) for c in range(self.numberOfCommunitiesAndNoise)}
        self.timeSeriesOfCommunities = timeSeriesOfCommunities
        self.initial_communities = initial_communities
        
    def generate_graphs_paths(self,number_of_steps,output_type=list):
        
        #time series full 
        fullTimeSeriesStack = np.array([self.timeSeriesOfCommunities[c] for c in range(0,self.numberOfCommunitiesAndNoise)])
        fullTimeSeriesStackCum = fullTimeSeriesStack.cumsum(axis=1)

        fullTimeSeries = fullTimeSeriesStack.sum(axis=0)
        cumFullTimeSeries = fullTimeSeries.cumsum()
        #################################################################

        snap_graphs = {0:snap_handlers.nx_to_snap(self.initial_graph)} # full graph
        graph_series = {}
        initial_relabeling = {c:dict(zip(range(len(self.initial_communities[c])),self.initial_communities[c])) \
                              for c in range(1,self.numberOfCommunitiesAndNoise)}
        
        # full graph
        graph_series[0] = [snap_handlers.snap_to_nx(snap_graphs[0])]
        for c in range(1,self.numberOfCommunitiesAndNoise):
            snap_graphs[c] = snap_handlers.nx_to_snap(communityGraphForEvolution(self.initial_graph,
                                                                                 self.initial_communities[c])) 
            graph_series[c] = [snap_handlers.snap_to_nx(snap_graphs[c])]
        relabeling = self.communityRelabelingForForestFire(number_of_steps)
        
        #HERE WE EVOLVE THE COMMUNITIES SEPARATLY
        for c in range(1,self.numberOfCommunitiesAndNoise):
            numberOfNodes = self.timeSeriesOfCommunities[c][0]
            if number_of_steps == len(self.timeSeriesOfCommunities[c]):
                if output_type == list:
                    for time, number_of_new_nodes in enumerate(self.timeSeriesOfCommunities[c][1:]):
                        numberOfNodes += number_of_new_nodes
                        self.forestFireModels[c].SetGraph(snap_graphs[c])
                        self.forestFireModels[c].AddNodes(int(numberOfNodes), True) #HERE IS THE EVOLUTION <<----------
                        new_networkx_graph = snap_handlers.snap_to_nx(snap_graphs[c])
                        graph_series[c].append(new_networkx_graph)
            else:
                print "Number of steps for series not match nodes time series"
                raise Exception
        
        #HERE WE EVOLVE THE FULL GRAPH
        if number_of_steps == len(self.timeSeriesOfCommunities[0]):
            if output_type == list:
                #after checking for consistency, we start the time series loop
                for time, number_of_new_nodes in enumerate(self.timeSeriesOfCommunities[0][1:]):
                    full_nx_graph  = snap_handlers.snap_to_nx(snap_graphs[0])
                    
                    # update the edges from the communities
                    for c in range(1,self.numberOfCommunitiesAndNoise):
                        new_community_graph = graph_series[c][time]
                        
                        relabeled_graph = nx.relabel_nodes(new_community_graph,relabeling[c][time-1])
                        relabeled_graph = nx.relabel_nodes(relabeled_graph,initial_relabeling[c])
                        full_nx_graph.add_edges_from(relabeled_graph.edges())
                        
                    numberOfNodes = cumFullTimeSeries[time+1] 
                                    
                    snap_graphs[0] = snap_handlers.nx_to_snap(full_nx_graph)
                    numberOfNodes += number_of_new_nodes
                    self.forestFireModels[0].SetGraph(snap_graphs[0])
                    self.forestFireModels[0].AddNodes(int(numberOfNodes), True)
                    
                    graph_series[0].append(snap_handlers.snap_to_nx(snap_graphs[0]))
                    
        else:
            print "Number of steps for series not match nodes time series"
            raise Exception

        self.full_membership = aggregateRelabelingForFullCommunityMembership(graph_series[0][-1],initial_relabeling,relabeling)
        
        return graph_series, relabeling, initial_relabeling
    
    
    def inference_on_graphs_paths(self,graphs_paths,output_type,dynamical_process):
        return None
    
    #==============================================
    # UTILS FOR THE CLASS
    #==============================================
    def communityRelabelingForForestFire(self,numberOfSteps):
        """
        This function provides the dictionary for relabeling the nodes during the dynamics of the commmunity forest fire
        
        Returns:
            relabeling
        """
        fullTimeSeriesStack = np.array([self.timeSeriesOfCommunities[c] for c in range(0,self.numberOfCommunitiesAndNoise)])
        fullTimeSeriesStackCum = fullTimeSeriesStack.cumsum(axis=1)
    
        fullTimeSeries = fullTimeSeriesStack.sum(axis=0)
        cumFullTimeSeries = fullTimeSeries.cumsum()
    
        relabeling = {c:[] for c in range(1,self.numberOfCommunitiesAndNoise)}
        for time in range(1,numberOfSteps):

            lower_border = int(cumFullTimeSeries[time-1])
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