import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

ylabel_dict = {'sd': 'suboptimal draws',
                'r': 'cumulative regrets',
                'bd': '% best arm drawn'
            }

# need to be modified
label_dict = {'False[4, 1]': 'HazardUCB',
              'UCB1_[1]': 'UCB1'}

line_style_list = ['-','--','-.',':']
marker_list = ['o',',','s','v','^', '.', '>', '<']
line_color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

arm_name_dict = {
    0: 'A',
    1: 'B',
    2: 'C'
}

LOW_CI_PER = 0.2
HIGH_CI_PER = 0.8

Plot_number_dict = {
    1: 'One',
    2: 'Two'
}

def plot_eva(results, eva_method, scale, compare_flag = True):
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
    plt.figure(figsize=(5 * 3, 5* len(results.keys())))
    
    for i, name in enumerate(results.keys()):
        plt.subplot(len(results.keys()),3, i+1)
        #plt.subplots(1,1, sharey = 'row')
        plt.title(name)
        plt.xlabel('iteration')
        plt.ylabel('expected ' + ylabel_dict[eva_method])
        for j, subname in enumerate(results[name].keys()):  
            low_CI = []
            high_CI = []
            if scale == 'raw':
                if subname != 'bound':
                    if not compare_flag:
                        if not subname.startswith('UCB'): 
                            
                            plt.plot(results[name][subname][eva_method], label = subname, color = line_color_list[j])
                            
                            #sns.lineplot(np.arange(1,1001), results[name][subname][eva_method])
                    else:
                        #data = results[name][subname][eva_method]
                        #sorted_data = sorted(data)
                        #low_CI.append(sorted_data[int(LOW_CI_PER * len(data))])
                        #high_CI.append(sorted_data[int(HIGH_CI_PER * len(data))])
                        plt.plot(range(len(results[name][subname][eva_method])), 
                                results[name][subname][eva_method], 
                                label = subname, 
                                color = line_color_list[j],  
                                marker = marker_list[j], 
                                markevery = 100,
                                markersize = 5)
            elif scale == 'log10':
                if subname != 'bound':
                    plt.plot(np.log10(np.asarray(results[name][subname][eva_method])), label = subname)
            #plt.fill_between(range(len(results[name][subname][eva_method])), low_CI, high_CI, color = line_color_list[j], alpha = 0.4)
        
        plt.legend()
    file_name = 'Exper_' + str(eva_method) + '.eps'
    #plt.savefig(file_name, bbox_inches='tight')

def plot_eva_for_paper(results, eva_method):
    """plot evaluations for paper

    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: dict
                keys: 'sd', 'r', 'bd'
                values: list of result  
    eva_method: str
        options ('sd', 'r', 'bd')
    """
    plt.figure(figsize=(5 * 3, 4* len(results.keys())))
    for i, name in enumerate(results.keys()):
        plt.subplot(len(results.keys()),3, i+1)
        #plt.subplots(1,1, sharey = 'row')
        if 'Outlier' in name:
            plt.title('With Outliers')
        else:
            plt.title('Without Outliers')
        plt.xlabel('Iterations')
        plt.ylabel('Expected ' + ylabel_dict[eva_method])
        for j, subname in enumerate(results[name].keys()): 
            label = subname.split('-')[0] 
            label = label.replace('_', '-')
            if subname != 'bound': 
                plt.plot(range(len(results[name][subname][eva_method])), 
                        results[name][subname][eva_method], 
                        label = label, 
                        color = line_color_list[j],  
                        marker = marker_list[j], 
                        markevery = 100,
                        markersize = 5)
        plt.legend(loc="upper left")
    file_name = 'Exper_' + str(eva_method) + '_' + name + '_' + subname + '.pdf'
    plt.savefig(file_name, bbox_inches='tight')

def plot_eva_for_clinical(results, eva_method):
    """plot evaluations for paper

    results: dict
        keys: 'env name + num_exper + num_rounds'
        values: dict
            keys: 'est_var + hyperpara' or 'bound'
            values: dict
                keys: 'sd', 'r', 'bd'
                values: list of result  
    eva_method: str
        options ('sd', 'r', 'bd')
    """
    print(len(results.keys()))
    plt.figure(figsize=(5 * 3, 4* len(results.keys())))
    for i, name in enumerate(results.keys()):
        plt.subplot(len(results.keys()),3, i+1)
        #plt.subplots(1,1, sharey = 'row')
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
            if subname != 'bound': 
                plt.plot(range(len(results[name][subname][eva_method])), 
                        results[name][subname][eva_method], 
                        label = label, 
                        color = line_color_list[j],  
                        marker = marker_list[j], 
                        markevery = 100,
                        markersize = 5)
        plt.legend(loc="upper left")
    file_name = data_name +'_treatmet.pdf'
    plt.savefig(file_name, bbox_inches='tight')

def plot_log_curve_fit(list_to_be_fit, name):
    xdata = range(1, len(list_to_be_fit)+1)
    popt, pcov = curve_fit(log_func, xdata, list_to_be_fit)
    plt.plot (xdata, list_to_be_fit, label = 'data')
    plt.plot(xdata, log_func(xdata, *popt), label = 'fit')
    plt.title(name + ' log curve fit')
    plt.legend()

def log_func(x, a, b, c):
    return a * np.log(b * x) + c

def plot_hist(sample_dict):
    '''plot hist for reward distribution samples 
    (to show what the distribution looks like)
    sample_dict: 
        key: e.g. 'AbsGau_[0.5, 0.1, 1.5]'
        value: list of list of samples
            len is the number of parameters (arms)
    '''
    plt.figure(figsize=(5 * 3, 5* len(sample_dict.keys())))
    #sns.set_style('darkgrid')
    j = 0
    for key, value in sample_dict.items():
        j += 1
        f = plt.subplot(len(sample_dict.keys()),3, j)
        plt.title(key)

        para_list = []
        split_list = key.split(']')
        for string in split_list:
            if string != '':
                a = string.split('[')[-1].replace(' ', '')
                para_list += list(a.split(','))
             
        for i, samples in enumerate(value):
            sns.distplot(samples, ax = f, label = para_list[i], bins = 100) 
            #f.axvline(np.median(samples), linestyle='-', label = 'median '+ para_list[1])
        plt.legend()

def plot_hist_for_paper(sample_dict):
    '''plot hist for reward distribution samples 
    (to show what the distribution looks like)
    sample_dict: 
        key: e.g. 'AbsGau_[0.5, 0.1, 1.5]'
        value: list of list of samples
            len is the number of parameters (arms)
    '''
    plt.figure(figsize=(5 * 3, 4* len(sample_dict.keys())))
    #sns.set_style('darkgrid')
    j = 0
    for key, value in sample_dict.items():
        j += 1
        f = plt.subplot(len(sample_dict.keys()),3, j)
        print(key)
        if 'Outlier' in key:
            plt.title('With Outliers')
        else:
            plt.title('Without Outliers')
        #plt.title('Group ' + Plot_number_dict[j])
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
            #f.axvline(np.median(samples), linestyle='-', label = 'median '+ para_list[1])
        plt.legend()
    file_name = 'Hist_'  + key + '.pdf'
    plt.savefig(file_name, bbox_inches='tight')

def plot_boxplot(sample_dict):
    '''
    sample_dict: 
        key: e.g. 'AbsGau_[0.5, 0.1, 1.5]'
        value: list of list of samples
            len is the number of parameters (arms)
    '''
    plt.figure(figsize=(5 * 3, 5* len(sample_dict.keys())))
    sns.set_style('darkgrid')
    counter = 0
    for key, value in sample_dict.items():
        counter += 1
        f = plt.subplot(len(sample_dict.keys()),3, counter)
        plt.title(key)

        para_list = []
        split_list = key.split(']')
        for string in split_list:
            if string != '':
                a = string.split('[')[-1].replace(' ', '')
                para_list += list(a.split(','))
        cato_list = []    
        sample_list = []
        for i in range(len(para_list)):
            for j in range(len(value[0])):
                cato_list.append(para_list[i])
                sample_list.append(value[i][j])
        df = pd.DataFrame({'cato': cato_list, 'sample': sample_list})
        #print(df.head())
        sns.boxplot(data = df, x = 'cato', y = 'sample', palette='Set3')