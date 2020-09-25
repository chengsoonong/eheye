import numpy as np

# Version: June/2020
# This file implements the simulated games for bandits algorithm. 

def simulate(env, summary_stat, policy, epsilon, m, budget, num_expers, 
            p, fixed_samples_list = None, tau = None):
    """Simulate best arm identification wrt specified summary statistics (tau-quantile, mean). 
    
    Paramters
    --------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    summary_stat: list
        sequence of summary statistics of arms, e.g. tau-quantile, mean, etc. 
    policy: object
        instance of Q_BAI, Mean_BAI
    epsilon: float
        accuracy level
    m: int
        number of arms to recommend
    budget: int 
        for fixed budget setting, budget (total number of rounds) as input
    num_expers: int
        total number of experiments
    p: object of ipywidgets.IntProgress
        show process bar when running experiments
    fixed_samples_list: list of dict, default is None
        each element is the sample dict for one exper
        key: arm_dix; values: list of fixed samples
    
   
    ------------Only for quantiles-------------------------------------
    tau: float
        quantile level
    """
    result = [] # list, element: 1 if simple regret bigger than epsilon (indicates error occurs);
                #                0 otherwise 

    for i in range(num_expers):
        p.value += 1
        if fixed_samples_list != None:
            samples = fixed_samples_list[i]
        else:
            samples = None

        if tau != None: # quantiles
            agent = policy(env, summary_stat, epsilon, tau, m, 
                    samples, budget)
        else: # mean
            agent = policy(env, summary_stat, epsilon, m, 
                    hyperpara, samples, udget_or_confi)
            est_H_flag, b
        agent.simulate()
        result.append(agent.evaluate())
        #if est_L_flag:
        #    estimated_L.append(agent.estimated_L_dict)
        
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