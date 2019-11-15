import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
    eva_dict =  defaultdict(dict)
    for key, values in sds.items():
        eva_dict[key]['sd'] = defaultdict(list)
        eva_dict[key]['sd'] = np.mean(np.asarray(values), axis = 0)
    for key, values in rs.items():
        eva_dict[key]['r'] = defaultdict(list)
        eva_dict[key]['r'] = np.mean(np.asarray(values), axis = 0)
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
    title_name = 'Cumulative Regret vs. Round'
    plt.title(title_name)

    plt.xlabel('Iteration')
    plt.ylabel('Expected ' + ylabel_dict[eva_method])
    
    for key, value in results.items():
        plt.plot(range(len(value[eva_method])), 
                value[eva_method],  
                marker = 'o', 
                markevery = 10,
                markersize = 5,
                label = key)

    plt.legend(loc="upper left")
    #file_name = 'Exper_' + str(eva_method) + '_' + name + '_' + subname + '.pdf'
    #plt.savefig(file_name, bbox_inches='tight')
