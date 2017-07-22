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

from graph_dynamics.networks import communities
from graph_dynamics.dynamics import GenerativeDynamicsCommunities
from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.dynamics import GraphsFormatsHandlers

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
        
        forest_fire_communities_parameters = {0:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.2,
                                                 "BackBurnProb":0.32,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.},
                                              1:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.7,
                                                 "BackBurnProb":0.72,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.},
                                              2:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.7,
                                                 "BackBurnProb":0.72,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.},
                                              3:{"BurnExpFireP":False,
                                                 "StartNNodes":1,
                                                 "ForwBurnProb":0.7,
                                                 "BackBurnProb":0.72,
                                                 "DecayProb":1.0,
                                                 "Take2AmbasPrb":0.,
                                                 "OrphanPrb": 0.}}
        
        numberOfCommunitiesAndNoise = len(forest_fire_communities_parameters.keys())
                
        numberOfSteps = 24
        #back ground evolution
        timeSeriesCommunity0 = np.ones(numberOfSteps)*2
        timeSeriesCommunity0[0] = 0
        
        timeSeriesCommunity1 = np.ones(numberOfSteps)*2
        timeSeriesCommunity1[0] = 30
        
        timeSeriesCommunity2 = np.ones(numberOfSteps)*3
        timeSeriesCommunity2[0] = 30
        
        timeSeriesCommunity3 = np.ones(numberOfSteps)*4
        timeSeriesCommunity3[0] = 20
        
        timeSeriesOfCommunities = {0:timeSeriesCommunity0,
                                   1:timeSeriesCommunity1,
                                   2:timeSeriesCommunity2,
                                   3:timeSeriesCommunity3}
        
        #the initial size of the community is that as defined by the time series
        numberOfNodesPerCommunities = [timeSeriesOfCommunities[c][0] for c in range(1,numberOfCommunitiesAndNoise)]
        numberOfBridgesPerCommunity = [1,1,1]
        barabasiParameter = 3
        
        initial_graph, subGraphs,Q,bridgesInCommunity = communities.barabasiAlbertCommunities(numberOfNodesPerCommunities, 
                                                              numberOfBridgesPerCommunity, 
                                                              barabasiParameter)
        initial_communities = {c:subGraphs[c-1].nodes() for c in range(1,numberOfCommunitiesAndNoise)}
        initial_communities[0]=[]
        
        print "Number of communities and Noise",numberOfCommunitiesAndNoise
        dynamics = GenerativeDynamicsCommunities.CommunitiesForestFire(initial_graph,
                                                                       initial_communities,
                                                                       forest_fire_communities_parameters,
                                                                       timeSeriesOfCommunities)

        graph_series, relabeling, initial_relabeling = dynamics.generate_graphs_paths(numberOfSteps)
        community_colors =  {c:colors[i] for i,c in enumerate(timeSeriesOfCommunities.keys())}
        
        fig, ax = plt.subplots(1,1,figsize=(24,12))
        graph_paths_visualization.plotGraphPathsCommunities(ax,graph_series[0],
                                                            dynamics.full_membership,
                                                            community_colors, 
                                                            series_name="community_ff")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.communitiesForestFireTest']
    unittest.main()