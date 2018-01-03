'''
Created on Jun 9, 2017

@author: cesar
'''
from scipy.stats import pareto, norm, bernoulli
import sys
import snap
import copy
import random
import numpy as np
from numpy import cumsum, sort, sum, searchsorted
from numpy.random import rand
import networkx as nx
import time
import math
from graph_dynamics.utils import snap_handlers
from graph_dynamics.networks.datatypes import VanillaGraph
from graph_dynamics.dynamics.datatypes import GraphsDynamics

#==========================================================================
# FOREST FIRE
#==========================================================================

class ForestFire(GraphsDynamics):
    """
    This is a wrapper for the snap function Forest Fire
    
    """
    def __init__(self, initial_graph,forestFireParameters,timeSeriesOfNodes,DYNAMICAL_PARAMETERS):
        """
        initial_graph: networkx graph
        
        BurnExpFireP: bool
        
        StartNNodes: int
        
        ForwBurnProb: double
        
        BackBurnProb: double
        
        DecayProb: double
        
        Take2AmbasPrb: double
        
        OrphanPrb: double
        
        timeSeriesOfNodes: numpy array
            the number of new nodes per time step
        """
        (BurnExpFireP,StartNNodes,ForwBurnProb,BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb) = forestFireParameters
        self.BurnExpFireP = BurnExpFireP
        self.StartNNodes = StartNNodes
        self.ForwBurnProb = ForwBurnProb
        self.BackBurnProb = BackBurnProb
        self.DecayProb = DecayProb
        self.Take2AmbasPrb = Take2AmbasPrb
        self.OrphanPrb = OrphanPrb
        
        type_of_dynamics = "snap_shot"
        self.dynamics_identifier = "ForestFire"
        self.timeSeriesOfNodes = timeSeriesOfNodes
        
        DYNAMICAL_PARAMETERS["DynamicsClassParameters"]={"ForestFire":forestFireParameters,
                                                         "TimeSeriesOfNodes":list(timeSeriesOfNodes)} 
        
        
        self.DYNAMICAL_PARAMETERS = DYNAMICAL_PARAMETERS
        self.Vanilla_0 =  VanillaGraph(self.dynamics_identifier,
                                       {"None":None},
                                       initial_graph)

        self.ff = snap.TFfGGen(BurnExpFireP,StartNNodes,ForwBurnProb,
                               BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb)
        
         
        GraphsDynamics.__init__(self, DYNAMICAL_PARAMETERS)
        
    def generate_graphs_paths(self,initial_graph,T):
        T = T - 1 
        print "initital generate paths ",initial_graph.get_networkx().number_of_nodes()
        initial_graph_nx = initial_graph.get_networkx()
        str_int = dict(zip(initial_graph_nx.nodes(),map(int,initial_graph_nx.nodes())))
        initial_graph_nx = nx.relabel_nodes(initial_graph_nx, str_int) 
        snap_graph = snap_handlers.nx_to_snap(initial_graph_nx)
        
        graph_series = [VanillaGraph(self.dynamics_identifier,
                                     {"None":None},
                                     snap_handlers.snap_to_nx(snap_graph))]
        numberOfNodes = graph_series[0].get_networkx().number_of_nodes()
        
        try:
            for i in range(0,int(T)):
                number_of_new_nodes = self.timeSeriesOfNodes[self.latest_index+i]
                numberOfNodes += number_of_new_nodes
                self.ff.SetGraph(snap_graph)
                self.ff.AddNodes(int(numberOfNodes), True)
                graph_series.append(VanillaGraph(self.dynamics_identifier,
                                                 {"None":None},
                                                 snap_handlers.snap_to_nx(snap_graph)))
        except:
            print sys.exc_info()
            print "Number of steps for series not match nodes time series"
            raise Exception
        
        return graph_series
    
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
 

#==========================================================================
# Activity Driven DYNAMICS
#==========================================================================

# Null model

class ActivityDrivenDynamics(GraphsDynamics):
    def __init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters):
        """
          Constructor

          Parameters

            TxGraph initial_graph:            initial state of a TxGraph instance
            int     number_of_connections:    max number of connections/edges a node can do

        """

        DYNAMICAL_PARAMETERS["DynamicsClassParameters"] = {"ActivityDrivenDynamics": None}
        # graph is a type of Acitivy Driven or Perra Graph
        self.initial_graph = initial_graph
        self.number_of_connections = extra_parameters["number_of_connections"]
        self.DYNAMICAL_PARAMETERS = DYNAMICAL_PARAMETERS
        self.extra_parameters = extra_parameters
        self.time_step = 0
        self.delta_in_seconds = extra_parameters["delta_in_seconds"]
        self.number_of_steps = DYNAMICAL_PARAMETERS["number_of_steps"]

        # activity
        self.activity_gamma     = extra_parameters["activity_gamma"]
        self.threshold_min      = extra_parameters["threshold_min"]
        self.number_of_nodes    = extra_parameters["number_of_nodes"]
        self.rescaling_factor   = extra_parameters["rescaling_factor"]
        self.delta_t            = extra_parameters["delta_t"]

        GraphsDynamics.__init__(self, DYNAMICAL_PARAMETERS)

        # ==================  set up the initial graph  ====================================================
        self.initial_graph.get_networkx().add_edges_from(self.initial_graph.get_networkx().edges(), {"time": 0})

        # ==================  set up the initial activity potential  =======================================
        self.activity_potential = self.init_activity_potential(self.activity_gamma, self.threshold_min, self.number_of_nodes)
        self.initial_graph      = self.set_activity(self.initial_graph, self.activity_potential)


        # if initial graph has no edges, do the first connection step
        if self.initial_graph.get_number_of_edges() == 0:
            before_connections = self.set_nodes_active(self.initial_graph)
            graph_after_connections = self.set_connections(before_connections)
            self.initial_graph = graph_after_connections
            # self.initial_graph.get_networkx().add_nodes_from(list(xrange(extra_parameters["number_of_nodes"])))



    # Abstract methods ====================================================

    def generate_graphs_paths(self, initial_graph, number_of_steps):
        """
          Method

          Parameters

            int     number_of_steps:   Total time steps to perform dynamics
            string  output_type:

        """
        graph_series = [self.initial_graph]
        for T in range(1, number_of_steps):
            graph_series.append(self.evolve_function(graph_series[T - 1]))

        return graph_series

    def set_graph_path(self):
        """
        Empirical Data
        """
        raise None

    def inference_on_graphs_paths(self, graphs_paths, output_type, dynamical_process=None):
        """
        Learning/Training
        """
        return None

    def get_dynamics_state(self):
        return self.DYNAMICAL_PARAMETERS

    def evolve_function(self, graph_state):
        """
        """

        # 0 clear connections
        graph_state.get_networkx().remove_edges_from(graph_state.get_networkx().edges())
        # 1 select nodes to be active
        before_connections = self.set_nodes_active(graph_state)
        # 2 make conenctions from activacted nodes
        graph_after_connections = self.set_connections(before_connections)

        # TODO: perra dynamics will handle the walker case
        # 3 make random walk
        # walked = self.__set_propagate_walker()
        # 4 change the acitivity base on the money  f(money)  = activity
        # 5 change the number of nodes and number of connects by function f(T) = # of nodes

        return copy.deepcopy(graph_after_connections)

    # Class methods ====================================================

    def init_activity_potential(self, activity_gamma, threshold_min, number_of_nodes):
        ## calculating the activity potential following pareto distribution
        X = pareto.rvs(activity_gamma, loc=threshold_min,size=number_of_nodes)  # get N samples from  pareto distribution
        X = X / max(X)  # every one smaller than one
        # return np.take(X, np.where(X > threshold_min)[0])  # using the thershold
        return X

    def set_activity(self,  graph_state, activity_potential):

        # run over all nodes to set initial attributes
        for n, node in enumerate(graph_state.get_networkx().nodes()):
            ## what is the purpose of rescaling factor?
            # ai = xi*n => probability per unit time to create new interactions with other nodes
            # activity_firing_rate is an probability number than [0,1]
            graph_state.get_networkx().node[n]['activity_firing_rate'] = activity_potential[n] * self.rescaling_factor
            # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
            graph_state.get_networkx().node[n]['activity_probability'] = graph_state.get_networkx().node[n]['activity_firing_rate'] * self.delta_t

        return graph_state

    def set_nodes_active(self, graph_state):
        for n in graph_state.get_networkx().nodes():
            graph_state.set_node_type(n)

        return graph_state

    def set_connections(self, graph_state):

        # list of choosed active nodes
        active_nodes = graph_state.get_active_nodes()
        # for each selected node make M connections
        for node in active_nodes:
            # 3-tuples (u,v,d) for an edge attribute dict d, or
            # select random M nodes to make M connection
            selected_nodes = [
                (node,
                 random.randint(0, graph_state.get_number_of_nodes() - 1),
                 {'time': random.randint(int(time.time()), int(time.time()) + self.delta_in_seconds) }
                 )
                 for e in range(self.number_of_connections)
                ]
            # make connections/edges

            # the connections are made as bucket and in our case each time connection step is a day in real life
            # we must simulate a day of connections by timestamp

            self.time_step = self.time_step + 1
            graph_state.get_networkx().add_edges_from(selected_nodes)

        return graph_state

class PerraDynamics(ActivityDrivenDynamics):

    def __init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters):

        self.initial_graph = initial_graph
        self.number_walkers = extra_parameters["number_walkers"]

        # TODO self.initial_graph is a type of perra graph

        ######################### initializing graph  #########################
        # run over all nodes to set initial attribute walker
        for n in self.initial_graph.get_networkx().nodes():
            self.initial_graph.get_networkx().node[n]['walker'] = 0

        # run over all nodes to set which nodes will start with a  walker
        generate_walkers = np.random.choice(self.initial_graph.get_networkx().nodes(), size=self.number_walkers,
                                            replace=False)
        for node in generate_walkers:
            self.initial_graph.add_walker(node)

        ActivityDrivenDynamics.__init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters)

    def set_propagate_walker(self, graph_state):

        walkers = graph_state.get_walkers()
        for node in walkers:
            # look at their neighbors: nodes that the walker is making an connection
            neighbors_nodes = graph_state.get_networkx().neighbors(node)

            if len(neighbors_nodes) > 0:
                # when a walker will not propagate he will stay at the same node
                neighbors_nodes.append(node)

                selected_neighbor = np.random.choice(neighbors_nodes, size=1, replace=False)
                selected_neighbor = selected_neighbor[0]

                graph_state.transfer_walker(_from=node, _to=selected_neighbor)

                if node != selected_neighbor:
                    print("walker  #" + str(node) + " moved to node #" + str(selected_neighbor))
                else:
                    print("walker node #" + str(node) + " did not move ")
            else:
                print("walker  #" + str(node) + " is trap, cant move because there is no node to go(edge)")

        return graph_state

    def evolve_function(self, graph_state):

        # 0 clear connections
        graph_state.get_networkx().remove_edges_from(graph_state.get_networkx().edges())
        # 1 select nodes to be active
        before_connections = self.set_nodes_active(graph_state)
        # 2 make conenctions from activacted nodes
        graph_after_connections = self.set_connections(before_connections)
        # 3 make random walk
        graph_state_walked = self.set_propagate_walker(graph_after_connections)

        # 4 change the acitivity base on the money  f(money)  = activity
        # 5 change the number of nodes and number of connects by function f(T) = # of nodes

        return copy.deepcopy(graph_state_walked)

# acitivity driven dynamics with walkers  described in the paper of Perra


# ==========================================================================
# BITCOIN DYNAMICS
# ==========================================================================



class BitcoinDynamics(GraphsDynamics):

    def __init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters):

        DYNAMICAL_PARAMETERS["DynamicsClassParameters"] = {"ActivityDrivenDynamics": None}
        # graph is a type of Acitivy Driven or Perra Graph


        self.number_of_connections  = extra_parameters["number_of_connections"]
        self.DYNAMICAL_PARAMETERS   = DYNAMICAL_PARAMETERS
        self.extra_parameters       = extra_parameters
        self.delta_in_seconds       = extra_parameters["delta_in_seconds"]
        self.number_of_nodes        = extra_parameters["number_of_nodes"]
        self.number_new_nodes       = extra_parameters["number_new_nodes"]
        self.number_of_steps        = DYNAMICAL_PARAMETERS["number_of_steps"]

        # activity
        # self.activity_gamma         = extra_parameters["activity_gamma"]
        # self.threshold_min          = extra_parameters["activity_threshold_min"]
        self.activity_rescaling_factor       = extra_parameters["activity_rescaling_factor"]
        self.activity_delta_t                = extra_parameters["activity_delta_t"]

        GraphsDynamics.__init__(self, DYNAMICAL_PARAMETERS)

        # ==================  set up the initial graph  ====================================================

        self.max_number_of_nodes = self.number_of_nodes + ((self.number_of_steps - 1) * self.number_new_nodes)

        # amount = self.init_amount(self.max_number_of_nodes ,extra_parameters["amount_pareto_gama"], extra_parameters["amount_threshold"])
        # initial_graph.set_amount(amount)

        initial_graph.get_networkx().add_edges_from(initial_graph.get_networkx().edges(), {"time": 0})
        self.initial_graph = initial_graph
        # # ==================  set up the initial activity potential  =======================================
        # activity_potential = self.init_activity_potential(self.max_number_of_nodes, extra_parameters["activity_gamma"],extra_parameters["activity_threshold_min"])
        # initial_graph.set_activity(activity_potential, self.activity_rescaling_factor, self.activity_delta_t )

        # ==================  set up the initial connections  =======================================
        # if initial graph has no edges, do the first connection step
        # if initial_graph.get_number_of_edges() == 0:
        #     before_connections      = self.set_nodes_active(initial_graph)
        #     graph_after_connections = self.set_connections(before_connections)
        #     self.initial_graph      = graph_after_connections
            # self.initial_graph.get_networkx().add_nodes_from(list(xrange(extra_parameters["number_of_nodes"])))


    # Abstract methods ====================================================

    def generate_graphs_paths(self, initial_graph, number_of_steps):
        """
          Method

          Parameters

            int     number_of_steps:   Total time steps to perform dynamics
            string  output_type:

        """
        graph_series = [self.initial_graph]
        for T in range(1, number_of_steps):
            graph_series.append(self.evolve_function(graph_series[T - 1]))

        return graph_series

    def set_graph_path(self):
        """
        Empirical Data
        """
        raise None

    def inference_on_graphs_paths(self):
        """
        Learning/Training
        """
        return None

    def get_dynamics_state(self):
        return self.DYNAMICAL_PARAMETERS

    def evolve_function(self, graph):

        # gs = graph.get_graph_state()
        graph.test = "1"

        # 0 clear connections
        graph.get_networkx().remove_edges_from(graph.get_networkx().edges())
        # 1 select nodes to be active
        graph.set_nodes_active()
        # 2 make conenctions from activacted nodes
        graph.set_connections(number_of_connections=self.number_of_connections, delta_in_seconds=self.delta_in_seconds)
        # 3 change the acitivity base on the money  f(money)  = activity
        graph.recalculate_activity_potential()
        # 4 change the number of nodes and number of connects by function f(T) = # of nodes
        graph.add_new_nodes(number_new_nodes=self.number_new_nodes)
        # 5 update the graph state
        graph.update_graph_state()

        return copy.deepcopy(graph)



    # Class methods ====================================================



class BitcoinMemoryDynamics(BitcoinDynamics):

    def __init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters):


        self.memory_number_of_connections = extra_parameters["memory_number_of_connections"]

        BitcoinDynamics.__init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters)


    # Abstract methods ====================================================

    def evolve_function(self, graph):

        # 0 clear connections
        graph.get_networkx().remove_edges_from(graph.get_networkx().edges())

        # 1 select nodes to be active
        graph.set_nodes_active()

        graph.set_nodes_memory_active()

        # 2 make conenctions from activacted nodes
        graph.set_connections(number_of_connections=self.number_of_connections, delta_in_seconds=self.delta_in_seconds)

        if (graph.memory_size > 0):
            graph.set_memory_connections(memory_number_of_connections=self.memory_number_of_connections, delta_in_seconds=self.delta_in_seconds)

        # 3 change the acitivity base on the money  f(money)  = activity
        graph.recalculate_activity_potential()
        graph.recalculate_memory_activity_potential()

        # 4 change the number of nodes and number of connects by function f(T) = # of nodes
        new_nodes = graph.add_new_nodes(number_new_nodes=self.number_new_nodes)
        graph.add_new_memory_nodes(new_nodes)

        # 5 update the graph state
        graph.update_graph_state()

        return copy.deepcopy(graph)














