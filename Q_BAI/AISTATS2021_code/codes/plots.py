import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Version: June/2020
# This file implements the plots methods for bandits algorithm. 

ylabel_dict = {'pe': 'probability of error',
               'sc': 'sample complexity',
            }
arm_name_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

line_style_list = ['-','--','-.',':']
marker_list = ['o','s','v','^', '.', '>', '<']
line_color_list = ['C0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
est_L_labels = ['Estimated L', 'L_at_10', 'L_at_200', 'True L']

#--------------------------------------------------------------------------------------

def plot_eva(results, eva_method, type = 'barplot', paper_flag = False, with_para = True, log_scale= False, 
            plot_confi_interval = False, method = 'all', exper = 'all', 
            title = 'Performance on simulated distributions', save_flag = True):
    """Plot method for evaluations

    Parameters
    -----------------------------------------------------------
    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: dict
                list of result  
    eva_method: str
        options ('pe', 'sc')
    type: str
        options ('barplot', 'lineplot')
    paper_flag: boolean
        indicates whether plotting for paper.
        True is for paper; False is not.
    log_scale: boolean
        if True, plot y axis as log scale
    plot_confi_interval: boolean
        if True, plot confidence interval (mu - sigma, mu + sigma))
    method: string
        if 'all', plot for all availble methods
        otherwise, plot specified method
    exper: string
        if 'all', plot for general uses
        otherwise, plot for specific format, e.g. 'est_L', 'hyperpara'
    save_flag: boolean
        True: save fig
    """
    fig = plt.figure(figsize=(4 * 3, 3* len(results.keys())))
    
    for i, name in enumerate(results.keys()):
        ax = fig.add_subplot(len(results.keys()),3, i+1)
        
        ax.set_title(title.replace('_', ' '))
        ax.set_xlabel('Algorithms')
        ax.set_ylabel(ylabel_dict[eva_method])
        
        for j, subname in enumerate(results[name].keys()):  
            
            # setup label
            if paper_flag:
                label = subname.replace('Adp-Q', 'Adp_Q').split('-')[0] 
                
                # change presented names
                if label == 'uniform_sampling':
                    label = 'Q-Uniform'
                if label == 'batch_elimination':
                    label = 'Q-BS'
                if label == 'Q_SAR_Simplified':
                    label = 'Q_SAR'
                if label == 'SAR_Simplified':
                    label = 'SAR'
                
                if with_para:
                    para = subname.split('-')[-1]
                    if ',' in  para:
                        label = label +  '-' + para.split(',')[0] + ']'
                    
                label = label.replace('_', '-')
            else: 
                label = subname

            if label == 'epsilon-greedy':
                label = r'$\epsilon$-greedy'

            if exper == 'est_L':
                label = est_L_labels[j]
                ax.set_title("Sensitiveness test for lower bound of hazard rate")
            elif exper == 'hyperpara':
                label = label.split(']')[0] + ']'

            if method == 'all' or (method !='all' and subname == method):

                mean = np.mean(results[name][subname])
                sigma = 0.1 * np.std(results[name][subname])

                if type == 'barplot':
                    if eva_method == 'pe':
                        width = 0.8
                    else:
                        width = 0.6
                    ax.bar([label], mean, width=width, yerr = sigma)

            plt.xticks(rotation=90)
    if log_scale:
        ax.set_yscale('log')
    if save_flag:
        file_name = '../plots/' + title + '.pdf'
        fig.savefig(file_name, bbox_inches='tight')

def plot_eva_m(results, eva_method, type = 'lineplot', paper_flag = False, log_scale= False, 
                plot_confi_interval = False, method = 'all', exper = 'all'):
    """Plot method for evaluations for choosing different number of m (recommended arms).

    Parameters
    -----------------------------------------------------------
    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: list of list
                list of results for m = 2 .. K-1 
    eva_method: str
        options ('pe', 'sc')
    type: str
        options ('barplot', 'lineplot')
    paper_flag: boolean
        indicates whether plotting for paper.
        True is for paper; False is not.
    log_scale: boolean
        if True, plot y axis as log scale
    plot_confi_interval: boolean
        if True, plot confidence interval (mu - sigma, mu + sigma))
    method: string
        if 'all', plot for all availble methods
        otherwise, plot specified method
    exper: string
        if 'all', plot for general uses
        otherwise, plot for specific format, e.g. 'est_L', 'hyperpara'
    """
    fig = plt.figure(figsize=(4 * 3, 3* len(results.keys())))
    
    for i, name in enumerate(results.keys()):
        ax = fig.add_subplot(len(results.keys()),3, i+1)
        
        ax.set_title("Performance on simulated distributions")
        ax.set_xlabel('m')
        ax.set_ylabel(ylabel_dict[eva_method])
        
        for j, subname in enumerate(results[name].keys()):  

            # setup label
            if paper_flag:
                label = subname.split('-')[0] 
                label = label.replace('_', '-')
            else: 
                label = subname

            if label == 'epsilon-greedy':
                label = r'$\epsilon$-greedy'

            if exper == 'est_L':
                label = est_L_labels[j]
                ax.set_title("Sensitiveness test for lower bound of hazard rate")
            elif exper == 'hyperpara':
                label = label.split(']')[0] + ']'

            if method == 'all' or (method !='all' and subname == method):
                mean_list = []
                std_list = []
                for result in results[name][subname]:
                    mean_list.append(np.mean(result))
                    std_list.append(np.std(result))
                if type == 'lineplot':
                    ax.plot(range(2, len(mean_list) + 2), mean_list, label=label)
    
            ax.legend()
    file_name = 'Exper_' + str(eva_method) + '_' + name + '_' + subname + '.pdf'
    # fig.savefig(file_name, bbox_inches='tight')


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

        plt.xlim([-0.1, 5])
        # plt.ylim([0,0.5])
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