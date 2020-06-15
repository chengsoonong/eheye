import math
import numpy as np

def update_q(q, x, y, tau, N, alpha = 1):
    pinball_loss = tau if x > q else (tau-1)
    q += ((N)*y + pinball_loss)*alpha/N
#     print ((N)*y + pinball_loss)
    y = pinball_loss
    return q, y


def update_q_lst(q_lst, tau_lst, y_lst, x, N, changing_alpha = True):

    alpha_lst = np.round([tau if tau<= 0.5 else 1-tau for tau in tau_lst], 4)*16
    if not changing_alpha: alpha_lst = np.ones(len(tau_lst)) * 0.5 *16
#     alpha_lst = np.ones(len(tau_lst)) * 0.5
    for i, q in enumerate(q_lst):
        tau = tau_lst[i]
        y = y_lst[i]
#         alpha = tau if tau<= 0.5 else 1-tau
        alpha = alpha_lst[i]
        q, y = update_q(q, x, y, tau, N, (1/(alpha)))
        q_lst[i] = q
        y_lst[i] = round(y, 5)
    return q_lst, y_lst


def get_SAG_procs(dt, tau_lst, changing_alpha = True):
    N = dt.shape[0]
    print ("--- SAG", changing_alpha)
    q_procs = np.zeros((N, len(tau_lst)))
    print (q_procs.shape)
    q_lst = np.zeros(len(tau_lst))
    y_lst = np.zeros(len(tau_lst))
    for i, x in enumerate(dt):
        q_lst, y_lst = update_q_lst(q_lst, tau_lst, y_lst, x, N, changing_alpha)
        q_procs[i]= q_lst
    return q_procs.T