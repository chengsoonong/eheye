import numpy as np

def L1_error(V, V_):
    return np.sum(np.absolute(V - V_))

def L2_error(V, V_):
    return np.sum(np.sqrt(np.sum(np.square(V - V_), axis=1)))

def Fisher_error(V, V_):
    return 2*np.trace(np.arccos(np.sqrt(V)@np.sqrt(V_.T)))

def symKL_error(V, V_):
    return np.sum((V-V_) * np.log(np.divide(V, V_)))

def report_evaluation(V_obs, V_fit):
    print("Fisher-Rao:   {:.5f}".format(Fisher_error(V_obs, V_fit)))
    print("Symmetric KL: {:.5f}".format(symKL_error(V_obs, V_fit)))
    print("L2 error:     {:.5f}".format(L2_error(V_obs, V_fit)))
    print("L1 error:     {:.5f}".format(L1_error(V_obs, V_fit)))
