import matplotlib.pyplot as plt
import numpy as np

ylabel_dict = {'sd': 'suboptimal draws',
                'r': 'cumulative regrets',
            }

def evaluate(sds, rs):
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
        eva_sd = np.mean(sds, axis = 0)
        eva_dict['sd'] = eva_sd
    if len(rs) > 0:
        rs = np.asarray(rs)
        eva_r = np.mean(rs, axis = 0)
        eva_dict['r'] = eva_r
    return eva_dict

def plot_eva(results, eva_method, paper_flag = True):
    """Plot method for evaluations

    Parameters
    -----------------------------------------------------------
    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: dict
                keys: 'sd', 'r'
                values: list of result  
    eva_method: str
        options ('sd', 'r')
    paper_flag: boolean
        indicates whether plotting for paper.
        True is for paper; False is not.
    """
    plt.figure(figsize=(5, 5))
       
    # setup title
    title_name = 'GPUCB with One-hot encoding'
    plt.title(title_name)

    plt.xlabel('Iteration')
    plt.ylabel('Expected ' + ylabel_dict[eva_method])
    
    plt.plot(range(len(results[eva_method])), 
            results[eva_method],  
            marker = 'o', 
            markevery = 10,
            markersize = 5)

    plt.legend(loc="upper left")
    #file_name = 'Exper_' + str(eva_method) + '_' + name + '_' + subname + '.pdf'
    #plt.savefig(file_name, bbox_inches='tight')

'''
 
'''