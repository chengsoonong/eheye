import numpy as np
import math
from Method_sgd_frugal_adaptive import get_adaptive_procs, get_frugal_procs, get_sgd_procs

def get_procs(dataset, tau_lst, method_name, **kwargs):
    if len(dataset.shape)!= 1: 
        raise Exception('Dataset for get_procs() of wrong shape:' + str(dataset.shape)+ ', should be 1d array')
    print ("get_procs() using method {}".format(method_name))

    method_dict = {
        'sgd': get_sgd_procs(dataset, tau_lst, **kwargs),
        'frugal': get_frugal_procs(dataset, tau_lst),
        'adaptive': get_adaptive_procs(dataset, tau_lst, **kwargs)
    }
    return method_dict.get(method_name)