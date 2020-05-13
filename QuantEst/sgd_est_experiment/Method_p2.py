import numpy as np


def extend_tau_lst(tau_lst):
    # extend tau_lst from [tau_1, ... tau_k] to 
    # [tau_1/2, tau_1, (tau_1 + tau_k)/2]
    k = len(tau_lst)
    help_lst = np.zeros(k+2)
    for i,x in enumerate(tau_lst):
        help_lst[i+1] = x
    help_lst[-1] = 1
    
    m = k*2+1
    ext_tau_lst = np.zeros(m)
    for i in range(len(ext_tau_lst)):
        k = int(i/2)
        if i%2 == 0:
            ext_tau_lst[i] = (help_lst[k] + help_lst[k+1])*0.5
        else:
            ext_tau_lst[i] = help_lst[k+1]
            
    ext_two_lst = np.zeros(m+2)
    ext_two_lst[0] = 0
    ext_two_lst[-1] = 1
    for i, t in enumerate(ext_tau_lst):
        ext_two_lst[i+1] = t
    
    return ext_two_lst

def find_k(x, q_lst):
    rtn = len(q_lst)-2
    for i, q in enumerate(q_lst):
        if x < q: 
            rtn = i-1
            break
    return max(0,rtn)
        
# find_k(18.8, np.arange(3, 20))

def adjust_ends(lst, x):
    lst[0] = min(lst[0], x)
    lst[-1] = max(lst[-1],x)
    return lst

# adjust_ends([-1,2,3,100], 200)

def update_marker_lst(marker_lst, k):
    for i in range(k+1, len(marker_lst)):
        marker_lst[i] += 1
    return marker_lst
# update_marker_lst([3,45,6,7,8], 3)

def p2(d, q, q_m, q_p, n, n_m, n_p):
    q_new = q + d/(n_p - n_m) \
        * ((n - n_m + d) * (q_p - q)/(n_p - n)\
         + (n_p - n - d) * (q - q_m)/(n - n_m))
    return q_new

# main function

def get_ext_p2_procs(dataset, tau_lst):
    m = 2*len(tau_lst) + 3
    
    # 0 = p1 < p2 < ... < pm = 1
    # m elements overall
    ext_tau_lst = extend_tau_lst(tau_lst)
    if (len(ext_tau_lst)!= m): raise Exception("len(ext_tau_lst)"+str(len(ext_tau_lst))+" m is "+str(m))
        
    # q1 < q2 < ... < q(m)
    # m elements overall
    ext_q_lst = np.sort(dataset[:m])
    min_x, max_x = ext_q_lst[0], ext_q_lst[-1]
    if (len(ext_q_lst)!= m): raise Exception("len(ext_q_lst)"+str(len(ext_q_lst))+" m is "+str(m))

    # current marker positions
    # n1 < n2 < ... < nm
    marker_lst = np.arange(1, m+1)
    
    # desired marker positions
    # n'1 < n'2 < ... < n'm
    # n'i = (N-1)*tau + 1, where N is the number of current observations so far
    desired_marker_lst = (m-1)*ext_tau_lst + 1
    ext_q_reco = np.zeros((len(dataset)-(m), m))
    
#     print (marker_lst, desired_marker_lst)
    
    for i, x in enumerate(dataset[m:]):
#         print ("data point number", i, ":", x)
        
        k = find_k(x, ext_q_lst)
        ext_q_lst = adjust_ends(ext_q_lst, x)
        # -------------- !!! 
        # notice the P2 paper got it wrong: 
        # in the algorithm "Box 1" PartB.2. Increment positions of markers k+1 through 5:
        # n_i += 1,   i = k,  ..., 5
        # should be   i = k+1,..., 5
        # otherwise the position of n_0 might change, which should never happen
        marker_lst = update_marker_lst(marker_lst, k)
        if marker_lst[-1] != i+1 + m or marker_lst[0] != 1: raise Exception("marker_lst wrong")
        desired_marker_lst = np.around(desired_marker_lst + ext_tau_lst, 4)

        #adjust heights of the inbetween markers n_1 ~ n_(m-2) if necessary
        for idx in range(1, m-1):
            n_p, n, n_m = marker_lst[idx+1], marker_lst[idx], marker_lst[idx-1]
            n_desire = desired_marker_lst[idx]
            d = round(n_desire - n, 2)
            
            if (d>=1 and n_p-n > 1) or (d<=-1 and n_m-n <-1):
                d = -1 if d<0 else 1
                q_m, q, q_p = ext_q_lst[idx-1], ext_q_lst[idx], ext_q_lst[idx+1] 
                
                # p2 adjustment
                q_new = p2(d, q, q_m, q_p, n, n_m, n_p)
                if q_m < q_new < q_p:
                    q = q_new
                    
                # linear adjustment
                else:
                    q2 = q_p if d>0 else q_m
                    n2 = n_p if d>0 else n_m
                    q = q + d * (q2-q)/(n2-n)
                
                ext_q_lst[idx] = q
                marker_lst[idx] = n+d
                
        ext_q_reco[i] = ext_q_lst
    return ext_q_reco

def get_p2_procs(dataset, tau_lst):
    # reshape the extended results to shape (m, N-m)
    ext_q_reco = get_ext_p2_procs(dataset, tau_lst).T

# ------------------------------------- Testing -------------------------------------

def get_original_lst(ext_lst):
    m = len(ext_lst)-2
    k = int((m-1)/2)
    lst = np.zeros(k)
    for i in range (1, m):
        if i % 2 == 0: 
            lst[int(i/2)-1] = ext_lst[i]
            
    return lst

# def test(tau_lst):
#     print (extend_tau_lst(tau_lst))
#     dt = np.random.normal(-20, 0.001, 2000)
# #     dt = np.append(dt, np.random.uniform(20, 60, 2000))
#     L = len((extend_tau_lst(tau_lst)))
#     q_true=np.percentile(dt, np.array(tau_lst) * 100)
#     q_lst= ext_p2(dt, tau_lst)
#     return q_true, q_lst.T

# tau_lst = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
# true_q, q_res = test(tau_lst)
# print (true_q)