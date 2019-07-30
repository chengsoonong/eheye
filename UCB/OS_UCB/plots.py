import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ylabel_dict = {'sd': 'suboptimal draws',
                'r': 'cumulative regrets',
                'bd': '% best arm drawn'
            }

def plot_eva(results, eva_method, scale):
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
    """
    plt.figure(figsize=(5, 4* len(results.keys())))
    for i, name in enumerate(results.keys()):
        plt.subplot(len(results.keys()),1, i+1)
        plt.title(name)
        plt.xlabel('iteration')
        plt.ylabel(scale + ' ' + ylabel_dict[eva_method])
        for subname in results[name].keys():  
            if scale == 'raw':
                #if subname != 'bound':
                plt.plot(results[name][subname][eva_method], label = subname)
            elif scale == 'log10':
                plt.plot(np.log10(np.asarray(results[name][subname][eva_method])), label = subname)
        plt.legend()
    #plt.savefig('tuning_beta.png')

def plot_log_curve_fit(list_to_be_fit, name):
    xdata = range(1, len(list_to_be_fit)+1)
    popt, pcov = curve_fit(log_func, xdata, list_to_be_fit)
    plt.plot (xdata, list_to_be_fit, label = 'data')
    plt.plot(xdata, log_func(xdata, *popt), label = 'fit')
    plt.title(name + ' log curve fit')
    plt.legend()

def log_func(x, a, b, c):
    return a * np.log(b * x) + c