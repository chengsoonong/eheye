import numpy as np

# Version: Feb/2020
# This file implements the simulated games for bandits algorithm. 

def simulate(env, summary_stat, policy, num_expers, num_rounds,
             est_var, hyperpara, fixed_L, p):
    """Simulate bandit games. 

    Paramters
    --------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    summary_stat: list
        sequence of summary statistics of arms, e.g. median, mean, etc. 
    policy: instance of UCB_discrete 
        one of M_UCB, UCB1, UCB-V, MV-LCB, Exp3
    num_expers: int
        total number of experiments
    num_rounds: int
        total number of rounds. 
    
    est_var: boolean, True indicates estimate parameters 
        e.g. lower bound of hazard rate L
    hyperpara: list of parameters
        parameters depending on different algorithms
    fixed_L: list (length: number of arms)
        if None, estimate L from samples; 
        otherwise, use fixed L (to test the sensitivity of the value of L)
    p: object of ipywidgets.IntProgress
        show process bar when running experiments
    """
    sds = []
    rs = []
    estimated_L = []

    for i in range(num_expers):
        p.value += 1
        agent = policy(env, summary_stat, num_rounds, hyperpara = hyperpara, est_flag = est_var, fixed_L = fixed_L)
        agent.play()
        sds.append(agent.suboptimalDraws)
        rs.append(agent.cumulativeSimpleRegrets)
        if hasattr(agent, 'estimated_L_dict'):
            estimated_L.append(agent.estimated_L_dict)

    eva_dict = evaluate(sds, rs, estimated_L)
    return eva_dict


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
    