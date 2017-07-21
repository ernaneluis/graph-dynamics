'''
Created on Jul 21, 2017

@author: cesar
'''


def createWindows(ALL_TIME_INDEXES,window,rolling):
    """
    indexes to be selected for analysis
    
    Parameters
    ----------
        ALL_TIME_INDEXES: list of ints
                no missing number is allowed
        window: int
        rolling: bool
    Returns
    -------
        WINDOWS
        [[int,...,],[int,...,],[int,...]]
        
    """
    maxA = max(ALL_TIME_INDEXES)
    minA = min(ALL_TIME_INDEXES)
    if ALL_TIME_INDEXES != range(minA,maxA+1):
        print "Missing graphs"
        raise Exception
    
    WINDOWS = []
    if not rolling:
        steps = range(minA,maxA,window)
        steps.append(maxA)
        for initial_window_index in steps[:-1]:
            window_list = range(initial_window_index,initial_window_index+window)
            if window_list[-1] <= maxA:
                WINDOWS.append(window_list)
    else:
        for a in ALL_TIME_INDEXES[:-window]:
            window_list = range(a,a+window+1)
            if window_list[-1] <= maxA:
                WINDOWS.append(window_list)
    return WINDOWS