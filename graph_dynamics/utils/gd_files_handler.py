'''
Created on Jul 17, 2017

@author: cesar
'''
import os
import sys


def copy_and_rename(gd_folder,dynamics_identifier):
    """
    """
    return None

def gd_folder_stats(gd_folder):
    """
    """
    ALL_DYNAMIC_FILES_NAME = os.listdir(gd_folder)
    
    GRAPH_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "gGD" in filename]
    STATE_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "sGD" in filename]
    MACRO_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "mGD" in filename]

    dynamics_identifier = GRAPH_FILES[0].split("_")[0]
    