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
    plt.figure(figsize=(5 * 3, 5* len(results.keys())))
    
    for i, name in enumerate(results.keys()):
        plt.subplot(len(results.keys()),3, i+1)
        
        # setup title
        if paper_flag:
            if 'Outlier' in name:
                title_name = 'With Outliers'
            else:
                title_name = 'No Outliers'
        else:
            title_name = name
        plt.title(title_name)

        plt.xlabel('Iteration')
        plt.ylabel('Expected ' + ylabel_dict[eva_method])
        
        for j, subname in enumerate(results[name].keys()):  

            # setup label
            if paper_flag:
                label = subname.split('-')[0] 
                label = label.replace('_', '-')
            else: 
                label = subname

            if (eva_method == 'r' and 'MV' not in label) or eva_method == 'sd':

                plt.plot(range(len(results[name][subname][eva_method])), 
                        results[name][subname][eva_method], 
                        label = label, 
                        color = line_color_list[j],  
                        marker = marker_list[j], 
                        markevery = 100,
                        markersize = 5)

            # control ylim, may need to adjust
            if eva_method == 'sd':
                plt.ylim([-10, 330])
            elif eva_method == 'r':
                plt.ylim([-10, 200])

        plt.legend(loc="upper left")
    file_name = 'Exper_' + str(eva_method) + '_' + name + '_' + subname + '.pdf'
    #plt.savefig(file_name, bbox_inches='tight')

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
    plt.figure(figsize=(5 * 3, 4* len(results.keys())))
    for i, name in enumerate(results.keys()):
        plt.subplot(len(results.keys()),3, i+1)
        data_name = name.split('_')[0]
        plt.title(data_name + ' Treatment Experiment')
        plt.xlabel('Iterations')
        plt.ylabel('Expected ' + ylabel_dict[eva_method])
        for j, subname in enumerate(results[name].keys()): 
            label = subname.split('-')[0] 
            label = label.replace('_', '-')
            if label == 'MV-LCB':
                if subname[10] == ',':
                    label+= '(50)'
                else:
                    label+= '(1e8)'
            if (eva_method == 'r' and 'MV-LCB' not in label)\
                or eva_method == 'sd':
                plt.plot(range(len(results[name][subname][eva_method])), 
                        results[name][subname][eva_method], 
                        label = label, 
                        color = line_color_list[j],  
                        marker = marker_list[j], 
                        markevery = 100,
                        markersize = 5)
            
            if eva_method == 'sd':
                plt.ylim([-20, 400])
            elif eva_method == 'r':
                plt.ylim([-1000, 20000])
                plt.ticklabel_format(style='sci', axis='y',  scilimits=(0,0))
        plt.legend(loc="upper left")
    file_name = data_name +'_treatmet_' + eva_method + '.pdf'
    #plt.savefig(file_name, bbox_inches='tight')


def plot_hist(sample_dict):
    '''plot hist for reward distribution samples.

    Parameters
    --------------------------------------------
    sample_dict: 
        key: e.g. 'AbsGau_[0.5, 0.1, 1.5]'
        value: list of list of samples
            len is the number of parameters (arms)
    '''
    plt.figure(figsize=(5 * 3, 5* len(sample_dict.keys())))
    j = 0
    for key, value in sample_dict.items():
        j += 1
        f = plt.subplot(len(sample_dict.keys()),3, j)
        if 'Outlier' in key:
            plt.title('With Outliers')
        else:
            plt.title('No Outliers')
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
        plt.legend()
    file_name = 'Hist_'  + key + '.pdf'
    #plt.savefig(file_name, bbox_inches='tight')

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