import numpy as np
import math
from Method_sgd_frugal_adaptive import get_adaptive_procs, get_frugal_procs, get_sgd_procs
from Method_p2 import get_p2_procs

def get_procs(dataset, tau_lst, method_name, **kwargs):
    if len(dataset.shape)!= 1: 
        raise Exception('Dataset for get_procs() of wrong shape:' + str(dataset.shape)+ ', should be 1d array')

    # print (method_name)
    method_dict = {
        'SGD' : get_sgd_procs,
        'Frugal': get_frugal_procs,
        'Adaptive': get_adaptive_procs,
        'P2': get_p2_procs,
    }
    # method_dict = {
    #     'SGD': get_sgd_procs(dataset, tau_lst, **kwargs),
    #     'Frugal': get_frugal_procs(dataset, tau_lst),
    #     'Adaptive': get_adaptive_procs(dataset, tau_lst, **kwargs),
    #     'P2': get_p2_procs(dataset, tau_lst),
    # }

    # if method_name in method_dict.keys():
    return  method_dict.get(method_name)(dataset, tau_lst, **kwargs)
    # else:
    #     raise Exception ("Method name {} is wrong! Please check the avaliable ones {}"
    #                     .format(method_name,
    #                     method_dict.keys()))