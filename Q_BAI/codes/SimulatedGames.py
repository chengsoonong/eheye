import numpy as np

# Version: Feb/2020
# This file implements the simulated games for bandits algorithm. 

def simulate(env, summary_stat, policy, epsilon, tau, m, budget_or_confi,
             num_expers, est_L_flag, hyperpara, fixed_L, p, fixed_samples_list = None, est_H_flag = False):
    """Simulate fixed budget BAI. 

    Paramters
    --------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    summary_stat: list
        sequence of summary statistics of arms, e.g. median, mean, etc. 
    policy: instance of UCB_discrete 
        one of M_UCB, UCB1, UCB-V, MV-LCB, Exp3
    epsilon: float
        accuracy level
    tau: float
        quantile level
    m: int
        number of arms to recommend
    budget_or_confi: int or float
        for fixed budget setting, budget (total number of rounds) as input
        for fixed confidence setting, confidence as input
    num_expers: int
        total number of experiments
   
    
    est_L_flag: boolean, True indicates estimate parameters 
        e.g. lower bound of hazard rate L
    hyperpara: list of parameters
        parameters depending on different algorithms
    fixed_L: list (length: number of arms)
        if None, estimate L from samples; 
        otherwise, use fixed L (to test the sensitivity of the value of L)
    p: object of ipywidgets.IntProgress
        show process bar when running experiments
    fixed_samples_list: list of dict, default is None
            each element is the sample dict for one exper
            key: arm_dix; values: list of fixed samples
    est_H_flag: boolean
        Only for QUGapEb
        True: est H
        False: use true H
        default is False
    """
    result = [] # list, element: 1 if simple regret bigger than epsilon (indicates error occurs);
                #                0 otherwise 
    estimated_L = []
    estimated_H = []


    for i in range(num_expers):
        p.value += 1
        if fixed_samples_list == None:
            agent = policy(env, summary_stat, epsilon, tau, m, 
                    hyperpara, est_L_flag, fixed_L, fixed_samples_list, est_H_flag, budget_or_confi)
        else:
            agent = policy(env, summary_stat, epsilon, tau, m, 
                    hyperpara, est_L_flag, fixed_L, fixed_samples_list[i], est_H_flag, budget_or_confi)
        agent.simulate()
        result.append(agent.evaluate())
        if hasattr(agent, 'estimated_L_dict'):
            estimated_L.append(agent.estimated_L_dict)
        if est_H_flag:
            estimated_H.append(agent.est_H_list)

    prob_error = np.mean(result)
    std = np.std(result)
    if est_H_flag:
        return result, estimated_H
    else:
        return result

def simulate_mean(env, summary_stat, policy, epsilon, m, budget_or_confi,
             num_expers, hyperpara, p, fixed_samples_list = None, est_H_flag = False):
    """Simulate fixed budget BAI. 

    Paramters
    --------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    summary_stat: list
        sequence of summary statistics of arms, e.g. median, mean, etc. 
    policy: instance of UCB_discrete 
        one of M_UCB, UCB1, UCB-V, MV-LCB, Exp3
    epsilon: float
        accuracy level
    m: int
        number of arms to recommend
    budget_or_confi: int or float
        for fixed budget setting, budget (total number of rounds) as input
        for fixed confidence setting, confidence as input
    num_expers: int
        total number of experiments
   
    hyperpara: list of parameters
        parameters depending on different algorithms
    p: object of ipywidgets.IntProgress
        show process bar when running experiments
    fixed_samples_list: list of dict, default is None
            each element is the sample dict for one exper
            key: arm_dix; values: list of fixed samples
    """
    result = [] # list, element: 1 if simple regret bigger than epsilon (indicates error occurs);
                #                0 otherwise 

    for i in range(num_expers):
        p.value += 1
        if fixed_samples_list == None:
            agent = policy(env, summary_stat, epsilon, m, 
                    hyperpara, fixed_samples_list, est_H_flag, budget_or_confi)
        else:
            agent = policy(env, summary_stat, epsilon, m, 
                    hyperpara, fixed_samples_list[i], est_H_flag, budget_or_confi)
        agent.simulate()
        result.append(agent.evaluate())

    prob_error = np.mean(result)
    std = np.std(result)
    return result

def evaluate(sds, rs, estimated_L_dict):
    """Calculated expected evaluation metrics (suboptimal draws, regret)

    Parameters
    ----------------------------------------------------
    sds: list 
        len: num_exper; list of sub-optimal draws for each experiment
    rs: list
        len: num_exper; list of cumulative regrets for each experiment

    Returns
    ----------------------------------------------------
    eva_dict: dict
        key: name of evaluation metrics
            one of 'sd' and 'r'
        value: array with shape 1 * num_rounds
    """
    eva_dict =  {}
    if len(sds) > 0:
        sds = np.asarray(sds)
        #eva_sd = np.mean(sds, axis = 0)
        #eva_dict['sd'] = eva_sd
        eva_dict['sd'] = sds # pass the whole list 
    if len(rs) > 0:
        rs = np.asarray(rs)
        #eva_r = np.mean(rs, axis = 0)
        #eva_dict['r'] = eva_r
        eva_dict['r'] = rs
    if len(estimated_L_dict) > 0:
        L = np.asarray(estimated_L_dict)
        eva_dict['estimated_L'] = L
    return eva_dict
    