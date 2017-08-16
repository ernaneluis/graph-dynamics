'''
Created on Jul 18, 2017

Each macro_file is defined as:

    <dynamicsidentifier>_mGD_<macroFileIdentifier>_<index>_.gd
    The basic statistics for the macros provided in a _gd directory
    are provided with the function:
    
    import graph_dynamics as gd
    gd.utils.gd_files_handler.gd_folder_stats(gd_directory)
    
    There are different types of macros
    
    scalars: {"macro":float,"macro2":float}
    node scalars: {"macro":{"node1":float,"node2":float,...,}}
    node vectors: {"macro":{"node1":array,"node2":array,...,}}
    
@author: cesar
'''
import sys
import json
import operator
import numpy as np
import pandas as pd
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics.datatypes import files_names


def get_time_series_index(DYNAMICS_PARAMETERS,macroNumbers,macrostate_file_indentifier):
    """
    """
    min_index = macroNumbers[macrostate_file_indentifier]['min_index']
    max_index = macroNumbers[macrostate_file_indentifier]['max_index']
    #TODO: logic related to datetime objects 
    #obtain index from DYNAMICS PARAMETERS 
    if DYNAMICS_PARAMETERS['datetime_timeseries']:
        initial_date =  DYNAMICS_PARAMETERS['initial_date']
        #dayfrequency = pd.date_range(start=minday,end=maxday , freq="{0}D".format(numberOfstepsInGraph))
        #dayfrequency = pd.date_range(start=minday,end=maxday , freq="{0}MS".format(numberOfstepsInGraph)
        pd.date_range(start=initial_date,size=len(max_index - min_index))
    else:
        return range(min_index,max_index + 1), min_index, max_index 
        
def time_index_macro(gd_directory,
                     macro_state_identifier,
                     macrostate_file_indentifier,
                     time_index):
    """
    Simply returns the macrostate by reading the file and selecting the json defined by the time_index: it selects one state/time of the time series
    
    macro_state_identifier: string
                    key of json which holds the desired macrostate
    macrostate_file_indentifier: string
                    file name
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,False)
    dynamics_foldername, graph_filename,graphstate_filename,macrostate_filename = files_names(DYNAMICS_PARAMETERS, 
                                                                                              time_index, 
                                                                                              macrostate_file_indentifier)
    macrostates = json.load(open(dynamics_foldername+macrostate_filename,"r"))[macro_state_identifier]
    return macrostates

def TS_dict_macro(gd_directory,
                  macro_state_identifier,
                  macrostate_file_indentifier,
                  macro_keys):
    """
    Parameters
    ----------
    gd_directory: string 
    macro_state_identifier:
    macrostate_file_indentifier:
    macro_keys: dict
    
    Returns
    -------
        pandas data frame with all time series
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,False)
    macro_keys = macro_keys.keys()
    macro_timeseries = {mk:[] for mk in macro_keys}
    
    #TODO: time series index with pandas
    #if not DYNAMICS_PARAMETERS["datetime_timeseries"]:
    for time_index in ALL_TIME_INDEXES:        
        dynamics_foldername, graph_filename,graphstate_filename,macrostate_filename = files_names(DYNAMICS_PARAMETERS, 
                                                                                                  time_index, 
                                                                                                  macrostate_file_indentifier)
        try:
            macrostates = json.load(open(dynamics_foldername+macrostate_filename,"r"))[macro_state_identifier]
            for mk in macro_keys:
                macro_timeseries[mk].append(macrostates[mk])
        except:
            print sys.exc_info()
            for mk in macro_keys:
                macro_timeseries[mk].append(None)
                
    return pd.DataFrame(macro_timeseries)

def TS_per_node_scalar_macro(gd_directory,macrostate_file_indentifier,macro_state_identifier,nodes=None,selectTop=5):
    """
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,False)
    time_index, min_index, max_index = get_time_series_index(DYNAMICS_PARAMETERS,macroNumbers,macrostate_file_indentifier)
    try:
        TS = np.zeros((len(time_index),len(nodes)))
    except:
        TS = np.zeros((len(time_index),selectTop))
        
    ALL_TIME_INDEXES.sort()    
    for j,file_index in enumerate(range(min_index,max_index+1)):        
        dynamics_foldername, graph_filename,graphstate_filename,macrostate_filename = files_names(DYNAMICS_PARAMETERS, 
                                                                                                  file_index, 
                                                                                                  macrostate_file_indentifier)
        try:
            macrostates = json.load(open(dynamics_foldername+macrostate_filename,"r"))[macro_state_identifier]
            if selectTop !=None:
                nodes = [a[0] for a in sorted(macrostates.iteritems(),key=operator.itemgetter(1))[::-1][:selectTop]]
            nodes_values = []
            for k in nodes:
                nodes_values.append(macrostates[k])
        except:
            print sys.exc_info()
            print "Macro read error, check file ",macrostate_filename
        TS[j] = np.array(nodes_values)
        
    return  pd.DataFrame(TS,index=time_index)


def TS_per_node_vector_macro(gd_directory,macrostate_file_indentifier,macro_state_identifier,node=0,selectTop=None,dimensions=5):
    """
    """
    return  None