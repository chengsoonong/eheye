import numpy as np

# Version: June/2020
# This file implements the simulated games for bandits algorithm. 

def simulate(env, summary_stat, policy, epsilon, tau, m, budget_or_confi, num_expers, 
             est_L_flag, hyperpara, fixed_L, p, fixed_samples_list = None, est_H_flag = False):
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

    return result   