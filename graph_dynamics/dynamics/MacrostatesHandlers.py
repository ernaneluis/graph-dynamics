'''
Created on Jul 18, 2017

@author: cesar
'''
from graph_dynamics.dynamics.datatypes import files_names
from graph_dynamics.utils import gd_files_handler
import pandas as pd
import numpy as np
import json

def TS_dict_macro(gd_directory,
                  macro_state_identifier,
                  macrostate_file_indentifier,
                  macro_keys):
    """
    Returns
    -------
        pandas data fram with all time series
    """
    ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,False)    
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
            for mk in macro_keys:
                macro_timeseries[mk].append(None)
                
    return pd.DataFrame(macro_timeseries)

def TS_of_node_scalar_macro():
    """
    """
    return None

def TS_of_node_vector_macro():
    """
    """
    return None
