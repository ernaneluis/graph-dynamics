import networkx as nx
import subprocess
import numpy as np
import os
class TemporalMotif():


    def __init__(self, GraphDynamics, input, delta):

        # exe_directory, gd_directory
        # name, delta
        # delta is the time window unit of time, if using timestamp than delta is in seconds

        self.exe_directory = "../../snap-cpp/examples/temporalmotifs/temporalmotifsmain"  # path of excecutable

        self.input_motif = self.graphdynamics_to_temporalmotif(GraphDynamics.get_networkx(), input)
        self.output_motif = self.input_motif.replace(".temporalmotif","") + ".temporalmotifcount"


        if os.path.isfile(self.output_motif ) == False:

            args1 = "-i:" + self.input_motif
            args2 = "-o:" + self.output_motif
            args3 = "-delta:" + str(delta)

            # calling command of snap in c++
            subprocess.call([self.exe_directory, args1, args2, args3])


    def get_motifdata(self):
        data = np.genfromtxt(self.output_motif, dtype=None)
        return data.flatten().tolist()

    def getKey(self,item):
        return item[2]["time"]

    def graphdynamics_to_temporalmotif(self, graph, input):
        # creating temporal graph file input

        output_path = input.replace(".gd", "") + ".temporalmotif"

        print "Converting graph dynamics to temporalmoitf output_path: " + output_path

        #if file dont exist convert it
        if os.path.isfile(output_path) == False:

            output_file = open(output_path, "w")

            set_nodes = set(graph.nodes())
            indexes = range(0, len(set_nodes))
            all_nodes = dict(zip(set_nodes, indexes))

            edges = graph.edges(data=True)
            edges = sorted(edges, key=self.getKey)

            for idy, edge in enumerate(edges):
                toa = all_nodes.get(edge[0])
                froma = all_nodes.get(edge[1])
                time = edge[2]['time']
                output_file.write(str(toa) + " " + str(froma) + " " + str(time) + "\n")

            output_file.close()
        print "Temporal File: " + output_path
        return output_path