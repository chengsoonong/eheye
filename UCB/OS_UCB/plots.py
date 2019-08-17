import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ylabel_dict = {'sd': 'suboptimal draws',
                'r': 'cumulative regrets',
                'bd': '% best arm drawn'
            }

# need to be modified
label_dict = {'False[4, 1]': 'HazardUCB',
              'UCB1_[1]': 'UCB1'}
def plot_eva(results, eva_method, scale, compare_flag):
    """
    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: dict
                keys: 'sd', 'r', 'bd'
                values: list of result  
    eva_method: str
        options ('sd', 'r', 'bd')
    scale: str
        'raw', 'log10'
    compare_flag: bool
        true: compare
        False: not compare
    """
    #plt.figure(figsize=(5 * 3, 5* len(results.keys())))
    for i, name in enumerate(results.keys()):
        #plt.subplot(len(results.keys()),3, i+1)
        plt.title(name)
        plt.xlabel('iteration')
        plt.ylabel(scale + ' ' + ylabel_dict[eva_method])
        for subname in results[name].keys():  
            if scale == 'raw':
                if subname != 'bound':
                    if not compare_flag:
                        if not subname.startswith('UCB'): 
                            plt.plot(results[name][subname][eva_method], label = subname)
                    else:
                        plt.plot(results[name][subname][eva_method], label = subname)
            elif scale == 'log10':
                if subname != 'bound':
                    plt.plot(np.log10(np.asarray(results[name][subname][eva_method])), label = subname)
        plt.legend()
    file_name = 'Exper_' + str(eva_method) + '.eps'
    #plt.savefig(file_name, bbox_inches='tight')

def plot_log_curve_fit(list_to_be_fit, name):
    xdata = range(1, len(list_to_be_fit)+1)
    popt, pcov = curve_fit(log_func, xdata, list_to_be_fit)
    plt.plot (xdata, list_to_be_fit, label = 'data')
    plt.plot(xdata, log_func(xdata, *popt), label = 'fit')
    plt.title(name + ' log curve fit')
    plt.legend()

def log_func(x, a, b, c):
    return a * np.log(b * x) + c