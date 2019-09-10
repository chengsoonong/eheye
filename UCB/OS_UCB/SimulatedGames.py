import numpy as np

# Options
# est_var (True, False), 
# evaluation (sub-optimal draws (sd), regret (r), % best arm draws (bd))
# hyperparameters: alpha, beta

def simulate(env, medians, policy, num_exper, num_rounds, est_var, 
            hyperpara, evaluation, p):
    """
    simulate multi-armed bandit games. 

    Paramters
    --------------------------------------------------------
    env: Environment object
    medians: list of medians of arms 
    policy: one of UCB_os_gau, UCB_os_exp, UCB1_os
    num_exper: int, number of independent experiments 
    num_rounds: int, number of total round for one game
    est_var: boolean, True indicates estimate variance
    hyperpara: list, [alpha, beta]
    evaluation: list of evaluation methods
                ['sd', 'r', 'bd']
    p: object of ipywidgets.IntProgress
        show process bar when running experiments
    """
    sds = []
    rs = []
    bds = []
    bound = {}

    for i in range(num_exper):
        p.value += 1
        agent = policy(env, medians, num_rounds, est_var, hyperpara, evaluation)
        agent.play()
        if 'sd' in evaluation:
            sds.append(agent.suboptimalDraws)
        if 'r' in evaluation:
            rs.append(agent.cumulativeRegrets)
        if 'bd' in evaluation:
            bds.append(agent.bestDraws)
    eva_dict = evaluate(sds, rs, bds, num_exper, num_rounds)
    bound['sd'] = agent.sd_bound()
    bound['r'] = agent.r_bound()
    #return eva_dict, bound, agent.estimated_para
    return eva_dict, bound


def evaluate(sds, rs, bds, num_exper, num_rounds):
    eva_dict =  {}
    if len(sds) > 0:
        #sd = np.asarray(sds).reshape((num_exper, num_rounds-3))
        sds = np.asarray(sds)
        eva_sd = np.mean(sds, axis = 0)
        eva_dict['sd'] = eva_sd
    if len(rs) > 0:
        #r = np.asarray(agent.cumulativeRegrets).reshape((num_exper, num_rounds-3))
        rs = np.asarray(rs)
        eva_r = np.mean(rs, axis = 0)
        eva_dict['r'] = eva_r
    if len(bds) > 0:
        #bd = np.asarray(agent.bestDraws).reshape((num_exper, num_rounds-3))
        bds = np.asarray(bds)
        eva_bd = np.mean(bds, axis = 0)
        eva_dict['bd']  = eva_bd
    return eva_dict
    