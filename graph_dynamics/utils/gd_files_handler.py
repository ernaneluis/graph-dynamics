'''
Created on Jul 17, 2017

@author: cesar
'''
import os
import sys
from shutil import copyfile


def copy_and_rename_graphs(gd_folder,old_dynamic_identifier,new_dynamics_identifier):
    """
    """
    old_dynamics_foldername =  gd_folder + old_dynamic_identifier + "_gd/"
    new_dynamics_foldername =  gd_folder + new_dynamics_identifier + "_gd/"
    
    #make sure folder for output is ready
    if not os.path.exists(new_dynamics_foldername):
        print "New Dynamics Directory"
        os.makedirs(new_dynamics_foldername)
    else:
        print "Dynamics Directory Exists"
        
    ALL_DYNAMIC_FILES_NAME = os.listdir(old_dynamics_foldername)
    GRAPH_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "gGD" in filename]
    ALL_TIME_INDEXES = [int(filename.split("_")[2]) for filename in GRAPH_FILES]
    
    for time_index in ALL_TIME_INDEXES:
        old_graph_filename = old_dynamics_foldername+"{0}_gGD_{1}_.gd".format(old_dynamic_identifier,time_index)
        new_graph_file_name  = new_dynamics_foldername+"{0}_gGD_{1}_.gd".format(new_dynamics_identifier,time_index) 
        #graphstate_filename = dynamics_foldername+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        copyfile(old_graph_filename, new_graph_file_name)
        

def gd_folder_stats(gd_folder):
    """
    """
    ALL_DYNAMIC_FILES_NAME = os.listdir(gd_folder)
    STATE_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "sGD" in filename]
    MACRO_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "mGD" in filename]
    GRAPH_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "gGD" in filename]
    ALL_TIME_INDEXES = [int(filename.split("_")[2]) for filename in GRAPH_FILES]
    
    return ALL_TIME_INDEXES, GRAPH_FILES 
