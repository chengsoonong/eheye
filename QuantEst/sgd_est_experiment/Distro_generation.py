import math
import numpy as np

distros = ['mix', 'gau_1', 'gau_2', 'exp']
stepsizes = ['const', '2_div_sqrt_k', '0.002_div_sqrt_k']

tau_vals = [0.1, 0.3, 0.5, 0.9, 0.99]
N_g = 12 # N_generation
N_s = 10 # N_shuffle

def get_mix_gauss(datasize):
    mix_lst = np.zeros(datasize)    
    sizes = np.array([0.3, 0.2, 0.1, 0.15, 0.25])
    mixtures = [(2,7), (0,0.7), (36, 26), (5,77), (-77,7)]
    acc_sizes = [sum(sizes[:i+1]) for i in range(len(sizes))]

    for d_idx in range(datasize):
        rdn = np.random.uniform(0,1)
        mix_id = 0
        for m_id in acc_sizes:
            if rdn > m_id:
                mix_id += 1
            else:break
        data_point = np.random.normal(mixtures[mix_id][0], mixtures[mix_id][1])
        mix_lst[d_idx] = data_point
    return mix_lst

def get_one_dt(distro, datasize):
#     return np.ones(size)
    if distro == 'gau_1':
        return np.random.normal(2, 18, datasize)
    elif distro == 'gau_2':
        return np.random.normal(0, 0.001, datasize)
    elif distro == 'mix':
        # mean: -1.3
        # std: 30.779035604224564
        # var: 947.3490327261234
        return get_mix_gauss(datasize)
    elif distro == 'exp':
        return np.random.exponential(scale=1, size=datasize)*6.5 - 20
    else: raise Exception("distribution doesn't work!")

def get_dataset(distro, datasize, g_test=False):
    if g_test:
        dataset = np.zeros((N_g, datasize))
        for i in range(N_g):
            dataset[i] = get_one_dt(distro, datasize)
    else:
        dataset = get_one_dt(distro, datasize)
    return dataset

def get_q_batch(dataset, tau_lst):
    if len(dataset.shape) != 1:
        raise Exception('Dataset for q_batch calculation of wrong shape: ' + str(dataset.shape))
        
    q_batch = np.zeros(len(tau_lst))
    for i, tau in enumerate(tau_lst):
        q_batch[i] = np.percentile(dataset, tau*100)
    return q_batch

def get_q_batches(dataset, tau_lst):
    # g_test = False
    if len(dataset.shape) == 1: 
        return get_q_batch(dataset, tau_lst)
    else:
        q_batches = np.zeros((dataset.shape[0], len(tau_lst)))
        for i in range(q_batches.shape[0]):
            q_batches[i] = get_q_batch(dataset[i], tau_lst)
    return q_batches

def get_q_true(distro, tau_lst):
    if tau_lst == tau_vals:
        if distro=='gau_1':
            return np.asarray([-21.06792817980280840537, 
                              -7.43920922874473411269,
                              2,
                              25.06792817980280840537,
                              43.87426173273513981594])
        elif distro=='gau_2':
            return np.asarray([-0.001281551565544600466965,
                              -5.244005127080407840383E-4,
                              0,
                              0.001281551565544600466965,
                              0.002326347874040841100886])
        elif distro=='mix':
            # sampled from 100000000 datapoints
            return np.asarray([-80.28496182,
                               -29.02324254,
                               -0.36011575,
                               36.69268923,
                               120.7676231])
        elif distro=='exp':
            return np.asarray([0.1053605156578263012275,
                              0.3566749439387323789126,
                              0.6931471805599453094172,
                              2.302585092994045684018,
                              4.605170185988091368036])*6.5 - 20
    else:
        dataset = get_dataset(distro, 1000000, True)
        q_batches = get_q_batches(dataset,tau_lst)
        return np.mean(q_batches, 0)

    raise Exception('tau_lst should be tau_vals')
