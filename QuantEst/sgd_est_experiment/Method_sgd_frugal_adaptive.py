import math
import numpy as np

#  ---------------------------------- Helper functions for all -----------------------------------
def set_sgd_stepsize(k, stepsize, length=None):
    if stepsize=='const':
        return 1 * np.ones(length) if not length else 1
    elif stepsize=='2_div_sqrt_k':
        return 2/math.sqrt(k)* np.ones(length) if not length else 2/math.sqrt(k)
    elif stepsize=='0.002_div_sqrt_k':
        return 0.002/math.sqrt(k) * np.ones(length) if not length else 0.002/math.sqrt(k)
    raise Exception('stepsize parameter is wrong', stepsize)
    
def sgd(q, x, tau, alpha):
    if x > q:
        q = q + alpha*tau
    else:
        q = q - alpha*(1-tau)  
    return q
    
def frugal(q, x, tau):
    rdn = np.random.uniform()
    if x > q and rdn > 1-tau:
        q += 1
    elif x < q and rdn > tau:
        q -= 1
    return q

# ---------------------------------------- get_sgd_procs ----------------------------------------

def get_sgd_procs(dataset, tau_lst, stepsize='const'):
    procs = np.zeros((len(tau_lst), dataset.shape[0]))
    for idx, tau in enumerate(tau_lst):
        q = 0
        q_sgd_proc = procs[idx]
        # change stepsize
        # if stepsize != 'frugal':
        for k, x in enumerate(dataset):
            alpha = set_sgd_stepsize(k+1, stepsize)
            # print (alpha)
            if x > q:
                q = q + alpha*tau
            else:
                q = q - alpha*(1-tau)
            q_sgd_proc[k] = q
    return procs

# ---------------------------------------- get_frugal_procs ----------------------------------------

def get_frugal_procs(dataset, tau_lst, **kwargs):
    procs = np.zeros((len(tau_lst), dataset.shape[0]))
    for idx, tau in enumerate(tau_lst):
        q = 0
        q_frugal_proc = procs[idx]
        for k, x in enumerate(dataset):
            rdn = np.random.uniform()
            if x > q and rdn > 1-tau:
                q += 1
            elif x < q and rdn > tau:
                q -= 1
            q_frugal_proc[k] = q
    return procs


# ---------------------------------------- get_adaptive_procs ----------------------------------------

def update_stepsize(alpha_arr, diff, step_size, update_size):
    update_var = np.ones(len(diff))
    for i, a in enumerate(alpha_arr):
        d = diff[i]
        if a * update_size * 0.25 < abs(d): update_var[i] = 2
        elif a * update_size * 0.02 > abs(d): update_var[i] = 1/2
#     print ('step_size', update_var)
    return update_var

def get_adaptive_procs(dataset, tau_lst, stepsize, update_size = 200):
    # print ("Adaptive: stepsize is {}".format(stepsize))
    if update_size*5 > dataset.shape[0]:
        print ("Warning!",
            "Cannot do the step size trick because the dataset of size {} is too small for the update size {}"
            .format(dataset.shape[0], update_size))
    proc = np.zeros((dataset.shape[0], len(tau_lst)))
    q_prev, q_now = np.zeros(len(tau_lst)), np.zeros(len(tau_lst))
    alpha_adaptation = np.ones(len(tau_lst))

    for k, x in enumerate(dataset):          
        if k % update_size == 0 and k >0:
#             print (k,q_now-q_prev)
            update_var = update_stepsize(alpha_arr, q_now-q_prev, stepsize, update_size)
#             print ('update_var', update_var)
            alpha_adaptation = alpha_adaptation * update_var
#             print ('alpha_adaptation', alpha_adaptation)
            q_prev = [i for i in q_now]
        if stepsize != 'frugal': 
            alpha_arr = set_sgd_stepsize(k+1, stepsize, len(tau_lst))* alpha_adaptation
#             if k % update_size == 0: print('alpha_arr', alpha_arr)
        for i, q in enumerate(q_now):
            tau = tau_lst[i]
            alpha = alpha_arr[i]
            if stepsize != 'frugal':
                q = sgd(q, x, tau, alpha)
            else: 
                q = frugal(q, x, tau)
            q_now[i] = q
        proc[k] = q_now
#     print (alpha_arr)
    return proc.T

