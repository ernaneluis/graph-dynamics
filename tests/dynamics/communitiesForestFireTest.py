'''
Created on Jul 5, 2017

@author: cesar
'''
import sys
sys.path.append("../../")

import json
import unittest
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_dynamics.networks.datatypes import VanillaGraph
from graph_dynamics.networks.communities import CommunityGraph

from graph_dynamics.networks import communities
from graph_dynamics.dynamics import GenerativeDynamicsCommunities
from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.dynamics import GraphsFormatsHandlers
from graph_dynamics.dynamics import datatypes

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True

#matplotlib.style.use('ggplot')

#matplotlib.style.use('seaborn-talk')
colors = []
for a in plt.style.library['bmh']['axes.prop_cycle']:
    colors.append(a["color"])
    
class Test(unittest.TestCase):
    
    def communitiesForestFireTest(self):                                                                                                
        number_of_steps = 100
        number_of_steps_in_memory = 1
                                                                                                                                                                       
        forest_fire_communities_parameters = {0:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.2,
                                                 "BackBurnProb":0.32,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.},                                                                                                                                                                                                                                                                                                                                                 
                                              1:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.2,
                                                 "BackBurnProb":0.32,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.},
                                              2:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.2,
                                                 "BackBurnProb":0.32,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.},
                                              3:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.2,
                                                 "BackBurnProb":0.32,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.}}
        
        numberOfCommunitiesAndNoise = len(forest_fire_communities_parameters.keys())
        
        #back ground evolution
        timeSeriesCommunity0 = np.ones(number_of_steps)
        timeSeriesCommunity0[0] = 0
        
        timeSeriesCommunity1 = np.ones(number_of_steps)*1
        timeSeriesCommunity1[0] = 10
        
        timeSeriesCommunity2 = np.ones(number_of_steps)*1
        timeSeriesCommunity2[0] = 10
        
        timeSeriesCommunity3 = np.ones(number_of_steps)*1
        timeSeriesCommunity3[0] = 10
        
        timeSeriesOfCommunities = {0:timeSeriesCommunity0,
                                   1:timeSeriesCommunity1,
                                   2:timeSeriesCommunity2,
                                   3:timeSeriesCommunity3}
        
        #the initial size of the community is that as defined by the time series
        numberOfNodesPerCommunities = [timeSeriesOfCommunities[c][0] for c in range(1,numberOfCommunitiesAndNoise)]
        numberOfBridgesPerCommunity = [2,2,2]
        barabasiParameter = 3
        initial_graph, subGraphs,Q,bridgesInCommunity = communities.barabasiAlbertCommunities(numberOfNodesPerCommunities, 
                                                                                              numberOfBridgesPerCommunity, 
                                                                                              barabasiParameter)
        initial_communities = {c:subGraphs[c-1].nodes() for c in range(1,numberOfCommunitiesAndNoise)}
        initial_communities[0]=[]
        
        
        simulations_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/"
        
        DYNAMICS_PARAMETERS = {"number_of_steps":number_of_steps,
                                "number_of_steps_in_memory":number_of_steps_in_memory,
                                "simulations_directory":simulations_directory,
                                "dynamics_identifier":"CommunityForestFire",
                                "graph_class":"CommunityGraph",
                                "verbose":True,
                                "datetime_timeseries":False,
                                "initial_date":1}
        
        DYNAMICS_PARAMETERS["macrostates"] =  [("basic_stats",())]
        
        vanilla_graph = VanillaGraph("Vanilla", 
                                     graph_state={"None":None}, 
                                     networkx_graph=initial_graph)
        
        community_graph = CommunityGraph(identifier_string="Communities",
                                         initial_comunities=initial_communities,
                                         networkx_graph=initial_graph)
        
        dynamics_object = GenerativeDynamicsCommunities.CommunitiesForestFire(community_graph,
                                                                              initial_communities,
                                                                              forest_fire_communities_parameters,
                                                                              timeSeriesOfCommunities,
                                                                              DYNAMICS_PARAMETERS)
        
        dynamics_object.evolve(10,community_graph)
        #graph_paths = dynamics_object.get_graph_path_window(1, 10)
        #nx_graph_paths = [g.get_networkx() for g in graph_paths]
        
        #nx_graph_series = [g.get_networkx() for g in GRAPH_SERIES]
        #COLORING AND PLOT
        #community_colors =  {c:colors[i] for i,c in enumerate(timeSeriesOfCommunities.keys())}
        #fig, ax = plt.subplots(1,1,figsize=(24,12))
        #graph_paths_visualization.plotGraphPathsCommunities(ax,
        #                                                    nx_graph_paths,
        #                                                    dynamics_object.full_membership,
        #                                                    community_colors, 
        #                                                    series_name="community_ff_{0}")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.communitiesForestFireTest']
    unittest.main()