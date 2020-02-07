import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

# Version: 25/Oct/2019
# This file implements the plots methods for bandits algorithm. 

# Mengyan Zhang, Australian National University; Data61, CSIRO.

ylabel_dict = {'sd': 'suboptimal draws',
                'r': 'cumulative regrets',
            }
arm_name_dict = {
    0: 'A',
    1: 'B',
    2: 'C'
}

line_style_list = ['-','--','-.',':']
marker_list = ['o','s','v','^', '.', '>', '<']
line_color_list = ['C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


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
    fig = plt.figure(figsize=(4 * 3, 3* len(results.keys())))
    
    for i, name in enumerate(results.keys()):
        ax = fig.add_subplot(len(results.keys()),3, i+1)
        
        # setup title
        if paper_flag:
            if 'Outlier' in name:
                title_name = 'With Outliers'
            else:
                title_name = 'No Outliers'
        else:
            title_name = name
        ax.set_title("Performance on simulated distributions")

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Expected ' + ylabel_dict[eva_method])
        
        for j, subname in enumerate(results[name].keys()):  

            # setup label
            if paper_flag:
                label = subname.split('-')[0] 
                label = label.replace('_', '-')
            else: 
                label = subname

            if label == 'epsilon-greedy':
                label = r'$\epsilon$-greedy'

            if (eva_method == 'r' and 'MV' not in label) or eva_method == 'sd':

                mean = np.mean(results[name][subname][eva_method], axis = 0)
                sigma = np.std(results[name][subname][eva_method], axis = 0)
                ax.plot(range(results[name][subname][eva_method].shape[1]), 
                        mean, 
                        #np.percentile(results[name][subname][eva_method], q=50, axis = 0),
                        label = label, 
                        color = line_color_list[j],  
                        marker = marker_list[j], 
                        markevery = 500,
                        markersize = 5)
                '''
                plt.fill_between(range(results[name][subname][eva_method].shape[1]), 
                        #np.percentile(results[name][subname][eva_method], q=30, axis = 0),
                        #np.percentile(results[name][subname][eva_method], q=70, axis = 0),
                        np.log(mean + sigma), np.log(np.max(mean - sigma, 0)),
                        color = line_color_list[j],  
                        alpha = 0.5)
                '''
                

            # control ylim, may need to adjust
            if eva_method == 'sd':
                #plt.ylim([-10, 330])
                ax.set_yscale('log')
                pass
            elif eva_method == 'r':
                #plt.ylim([-10, 230])
                pass

        ax.legend(loc="lower right")
    file_name = 'Exper_' + str(eva_method) + '_' + name + '_' + subname + '.pdf'
    fig.savefig(file_name, bbox_inches='tight')

def plot_eva_for_clinical(results, eva_method):
    """Plot method for clinical datasets.

    Parameters
    ----------------------------------------------
    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: dict
                keys: 'sd', 'r'
                values: list of result  
    eva_method: str
        options ('sd', 'r')
    """
    fig = plt.figure(figsize=(5 * 3, 4* len(results.keys())))

    for i, name in enumerate(results.keys()):
        ax = plt.subplot(len(results.keys()),3, i+1)
        data_name = name.split('_')[0]
        if data_name == "Glinoma":
            data_name = "Glioma"
        ax.set_title(data_name + ' Treatment Experiment')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Expected ' + ylabel_dict[eva_method])
        for j, subname in enumerate(results[name].keys()): 
            label = subname.split('-')[0] 
            label = label.replace('_', '-')
            if label == 'epsilon-greedy':
                label = r'$\epsilon$-greedy'
            if (eva_method == 'r' and 'MV-LCB' not in label)\
                or eva_method == 'sd':
                mean = np.mean(results[name][subname][eva_method], axis = 0)
                sigma = np.std(results[name][subname][eva_method], axis = 0)
                ax.plot(range(results[name][subname][eva_method].shape[1]), 
                        mean, 
                        #mean,
                        label = label, 
                        color = line_color_list[j],  
                        marker = marker_list[j], 
                        markevery = 500,
                        markersize = 5)
            
            
        #ax.set_ylim([10**-0.8, 10**3.5])
        ax.set_yscale('log')
        ax.legend(loc="lower right")
    file_name = data_name +'_treatmet_' + eva_method + '.pdf'
    plt.savefig(file_name, bbox_inches='tight')


def plot_hist(sample_dict):
    '''plot hist for reward distribution samples.

    Parameters
    --------------------------------------------
    sample_dict: 
        key: e.g. 'AbsGau_[0.5, 0.1, 1.5]'
        value: list of list of samples
            len is the number of parameters (arms)
    '''
    plt.figure(figsize=(4 * 3, 3 * len(sample_dict.keys())))
    j = 0
    for key, value in sample_dict.items():
        j += 1
        f = plt.subplot(len(sample_dict.keys()),3, j)
        '''
        if 'Outlier' in key:
            plt.title('With Outliers')
        else:
            plt.title('No Outliers')
        '''
        plt.title('Reward Histogram')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')

        para_list = []
        split_list = key.split(']')
        for string in split_list:
            if string != '':
                a = string.split('[')[-1].replace(' ', '')
                para_list += list(a.split(','))
             
        for i, samples in enumerate(value):
            sns.distplot(samples, ax = f, label = arm_name_dict[i], bins = 100, norm_hist=False) 

        plt.xlim([-1, 20])
        # plt.ylim([0, 0.4])
        plt.legend()
    file_name = 'Hist_'  + key + '.pdf'
    plt.savefig(file_name, bbox_inches='tight')

#-----------------------------------------------------------------------
# fit a log curve 

def log_func(x, a, b, c):
    return a * np.log(b * x) + c
    
def plot_log_curve_fit(list_to_be_fit, name):
    xdata = range(1, len(list_to_be_fit)+1)
    popt, pcov = curve_fit(log_func, xdata, list_to_be_fit)
    plt.plot (xdata, list_to_be_fit, label = 'data')
    plt.plot(xdata, log_func(xdata, *popt), label = 'fit')
    plt.title(name + ' log curve fit')
    plt.legend()