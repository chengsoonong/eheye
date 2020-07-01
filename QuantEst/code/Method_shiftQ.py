import numpy as np

# not sure why *q_est --> Cannot deal with q_est<0
def DUMIQUE(q_est, x, tau, stepsize):
    if q_est < 0: raise Exception("!!!", q_est)
    if x > q_est:
        q_est += (stepsize*q_est)*tau
    else:
        q_est += (stepsize*q_est)*(tau-1)
    return q_est

# # SGD which works for q_est non positive, only constant stepsize
# def sgd(q_est, x, tau, stepsize):
#     if x > q_est:
#         q_est += stepsize*tau
#     else:
#         q_est += stepsize*(tau-1)
#     return q_est

def checkinput(q_est):
    q_new = [-0.001]
    q_new.extend([q_est[i] for i in range(len(q_est)-1)])
    for i in range(len(q_est)):
        if q_new[i] >= q_est[i]: return False
    return True

# sX is stepsize_X, sY is stepsize_Y
# c = 2 because 0.5 is at tau_vals[2]
def shiftQ(dataset, tau_lst, sX, sY, qX, qY, c=0):
    if not (checkinput(qX) and checkinput(qY)):
        raise Exception("Input quantile estimate not applicable")
    count = 0
    proc = np.zeros((len(dataset), len(tau_lst)))
    for i, x in enumerate(dataset):
        qX[c] = DUMIQUE(qX[c], x, tau_lst[c], sX)

        for k in range(c-1, -1, -1):
            tau = tau_lst[k]
#           shifted observation
            yk = qX[k+1] - x
#           shifted Y distro
            qY[k] = DUMIQUE(qY[k], yk, 1-tau, sY)
#           shift back
            qX[k] = qX[k+1] - qY[k]
#             print ('yk', yk)
#             print ('qY', qY)
#             print ('qX', qX)
        
        for k in range(c+1, len(qX)):
            tau = tau_lst[k]
            yk = -qX[k-1] + x
            qY[k] = DUMIQUE(qY[k], yk, tau, sY)
            qX[k] = qX[k-1] + qY[k]
#             print ('yk',yk)
#             print ('qY', qY)
#             print ('qX', qX)
#         print ('qY', qY)
        if not (checkinput(qX)):
            count += 1
        proc[i] = qX
    print ("shiftQ Overall crossing", count)
    return proc.T

def get_shiftQ_procs(dataset, tau_lst, **kwargs):
    sX = 0.01
    sY = 0.02
    qX = np.asarray(tau_lst)
    qY = np.asarray(tau_lst)+1.0
    return shiftQ(dataset, tau_lst, sX, sY, qX, qY)


