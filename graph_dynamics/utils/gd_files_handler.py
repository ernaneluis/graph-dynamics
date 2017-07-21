'''
Created on Jul 17, 2017

@author: cesar
'''
import os
import sys
import json
import pprint
from shutil import copyfile

def parse(text):
    try:
        return json.loads(text)
    except ValueError as e:
        print('invalid json: %s' % e)
        return None # or: raise
    
def copy_and_rename_graphs(simulations_directory,old_dynamic_identifier,new_dynamics_identifier):
    """
    simulations_directory: place where the _gd folder is located
    """
    old_dynamics_foldername =  simulations_directory + old_dynamic_identifier + "_gd/"
    new_dynamics_foldername =  simulations_directory + new_dynamics_identifier + "_gd/"
    
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
        

def gd_folder_stats(gd_directory,vprint=False):
    """
    Basic statistics of the files
    
    If the DYNAMICS_PARAMETERS are not present in the gd_directory
    the function will provide a json object with keys:
    "dynamics_identifier","simulations_directory"
    
    Returns
    -------
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers
    
    ALL_TIME_INDEXES: list of int
    
    DYNAMICS_PARAMETERS: json with all the dynamics information requiered
    
    macroNumbers: dict 
                macroNumbers[macrostring] = {"size":int,
                                             "min_index":int,
                                             "max_index":int}
    """
    ALL_DYNAMIC_FILES_NAME = os.listdir(gd_directory)
    try:
        DYNAMICS_PARAMETERS = json.load(open(gd_directory+"DYNAMICS_PARAMETERS","r"))
        if vprint:
            print "Dynamics {0}".format(DYNAMICS_PARAMETERS["dynamics_identifier"])
            pprint.pprint(DYNAMICS_PARAMETERS)
    except:
        dynamics_identifier = gd_directory.split("/")
        dynamics_identifier = dynamics_identifier[-2].split("_")[0] 
        simulation_directory = "/".join(gd_directory.split("/")[:-2])+"/"
        DYNAMICS_PARAMETERS = {"dynamics_identifier":dynamics_identifier,"simulations_directory":simulation_directory}
        if vprint:
            print "NO DYNAMICS PARAMETERS, POSSIBLY EMPTY STATES FOLDER"
    
    STATE_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "sGD" in filename]
    MACRO_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "mGD" in filename]
    GRAPH_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "gGD" in filename]
    ALL_TIME_INDEXES = [int(filename.split("_")[2]) for filename in GRAPH_FILES]
    ALL_TIME_INDEXES.sort()
    
    macroStrings = set([macro_file.split("_")[2] for macro_file in MACRO_FILES])
    macroNumbers = {}
    for macrostring in macroStrings:
        this_macro_files = [filename for filename in ALL_DYNAMIC_FILES_NAME if macrostring in filename]
        indexes = [int(macrofile.split("_")[-2]) for macrofile in this_macro_files]
        example_macro_json = json.load(open(gd_directory+this_macro_files[-1],"r"))
        macro_keys = example_macro_json.keys()
        macroNumbers[macrostring] = {"size":len(this_macro_files),
                                     "min_index":min(indexes),
                                     "max_index":max(indexes),
                                     "macros":{mk:type(example_macro_json[mk]) for mk in macro_keys}}
        
    numberOfGraphFiles = len(GRAPH_FILES)
    numberOfStates = len(STATE_FILES)
    
    if vprint:
        print "Number of state files {0}".format(numberOfStates)
        print "Number of Graphs {0}".format(numberOfGraphFiles)
        print "Macros "
        pprint.pprint(macroNumbers)
    
    return ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers
