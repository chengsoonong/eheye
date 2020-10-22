import numpy as np
from collections import defaultdict
from scipy.special import erf
from sklearn.neighbors import KernelDensity


# Version: June/2020
# This file implements the environment construction for bandits algorithm.
# Functions: 
#   setup_env:  Setup environment for simulations.
#   generate_samples: Generate samples for each env (creating fixed sample list for debug)
#   est_L_test: Test of convergence of estimation of L.

# Simulated environments:
# mixture absolute Gaussian, Exponential

# ---------------------------------------------------------------------------------

def setup_env(envs_setting, ss_list = ['quantile_0.5'], random_set = None):
    """Setup environment for simulations.

    Parameter:
    --------------------------------------------------
    envs_setting: list of env dict
        keys: instance of environment class 
            (AbsGau, Exp, AbsGau_Outlier, Exp_Outlier)
        values: list of parameters
        e.g.environments = [
                {AbsGau: [0.5, 1.0, 1.5]}, 
                {Exp:    [2.0, 1.0, 1.5]},
                {AbsGau: [0.5], Exp: [1.0, 1.5]}
               ]
    ss_list: list of string
        list of summary statistics names, with the name and parameter

    Return:
    --------------------------------------------------
    results_env: dict of list
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of env instances
    tau-quantiles: dict of list for tau-quantiles
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of medians
    means: dict of list for means
    mvs: dict of list for mean-variance
    samples: dict of list for samples
    """
    
    rewards_env = defaultdict(list)
    samples = defaultdict(list)
    true_ss_dict = {}
    L = defaultdict(list) # lower bound of hazard rate
    num_samples = 50000

    for envs_dict in envs_setting:
        name = ''
        print(envs_dict)
        for env, para_list in envs_dict.items():
            env_name = str(env).split('.')[-1][:-2]
            name += env_name + '_' + str(para_list)
        true_ss_dict[name] = defaultdict(list)
       
        for env, para_list in envs_dict.items(): 
            for para in para_list:
                if random_set == None:
                    if type(para) is list:
                        current_env = env(*para)
                    else:
                        current_env = env(para)
                else:
                    current_env = env(para, random_set)
                rewards_env[name].append(current_env)
                sample = current_env.sample(num_samples)
                samples[name].append(sample)
                # the lower bound is at 0 for current environments (AbsGau, Exp)
                L[name].append(current_env.hazard_rate(0))
                
                # for ss in ss_list:
                #     ss_name = ss.split('_')[0]
                #     ss_para = float(ss.split('_')[-1])
                #     true_ss_dict[name][ss] = []
                for ss in ss_list:
                    ss_name = ss.split('_')[0]
                    if len(ss.split('_'))> 1:
                        ss_para = float(ss.split('_')[-1])
                    if ss_name == 'quantile':
                        true_ss_dict[name][ss].append(np.quantile(sample, ss_para))
                    elif ss_name == 'mean':
                        true_ss_dict[name][ss_name].append(np.mean(sample))
                    else:
                        assert True, 'Unknown summary statistics!'

    return rewards_env, true_ss_dict, samples, L

def generate_samples(envs_dict, tau =0.5, num_samples = 5000):
    """Generate samples for each env. 

    Parameter:
    --------------------------------------------------
    envs_dict: env dict
        keys: instance of environment class 
            (AbsGau, Exp, AbsGau_Outlier, Exp_Outlier)
        values: list of parameters
        e.g. {AbsGau: [[0.5, 1.0, 1.5]], Exp: [[1/4]]}, 
               
    tau: float (0,1)
        level of quantile
    num_samples: int
        number of samples for each env.

    Return:
    --------------------------------------------------
    samples: dict of list for samples
        key: arm/env idx
        value: list of sampes (size: num_sample) for arm idx
    """
    
    samples = defaultdict(list)

    arm_idx = 0

    for env, para_list in envs_dict.items(): 
        for para in para_list: 
            if type(para) is list:
                current_env = env(*para)
            else:
                current_env = env(para)
            
            sample = current_env.sample(num_samples)
            samples[arm_idx] = sample
            arm_idx += 1
    
    return samples

def est_L_test(envs_setting, num_exper = 100, sample_size = 500, est_method = 'naive', bandwidth = 1):
    """Test of convergence of estimation of L. 

    Parameter:
    --------------------------------------------------
    envs_setting: list of env dict
        keys: instance of environment class 
            (AbsGau, Exp, AbsGau_Outlier, Exp_Outlier)
        values: list of parameters
        e.g.environments = [
                {AbsGau: [0.5, 1.0, 1.5]}, 
                {Exp:    [2.0, 1.0, 1.5]},
                {AbsGau: [0.5], Exp: [1.0, 1.5]}
               ]
    sample_size: int
        number of samples
    est_method: str
        can be one of the choice of 'kde', 'naive'
        'kde': kernel density estimation
        'naive': count the samples insides [0,dt]
    bandwidth: float

    Return:
    --------------------------------------------------
    est_Ls: dict 
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of est Ls for each round
    """
    
    est_Ls = defaultdict(list)

    for envs_dict in envs_setting:
        #name = ''
        #for env, para_list in envs_dict.items():
            #env_name = str(env).split('.')[-1][:-2]
            #name += env_name + '_' + str(para_list)

        arm_idx = 0
        for env, para_list in envs_dict.items(): 
            for para in para_list:
                if type(para) is list:
                    current_env = env(*para)
                else:
                    current_env = env(para)

                #est_Ls[arm_idx] = []
                for exper in range(num_exper):
                    current_samples = []
                    current_est_Ls = []
                    for t in range(sample_size):
                        current_samples.append(current_env.sample())
                        current_est_Ls.append(est_L(current_samples, est_method, bandwidth))
                    est_Ls[arm_idx].append(current_est_Ls)
                arm_idx +=1
    return est_Ls

def est_L(sample_list, est_method, bandwidth = 0.5):
    """Estimate L from a list of samples.

    Parameter
    ------------------------------------------
    sample_list: list
        a list of samples for arm i at time t
    est_method: str
        can be one of the choice of 'kde', 'naive'
        'kde': kernel density estimation
        'naive': count the samples insides [0,dt]
    """
    if est_method == 'kde':
        kde = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(np.asarray(sample_list)[:, np.newaxis])
        log_den_0 = kde.score_samples(np.asarray([0])[:, np.newaxis])
        estL = np.exp(log_den_0)[0]
    elif est_method == 'naive':
        sorted_data = np.asarray(sorted(sample_list))
        estL = len(sorted_data[sorted_data <= bandwidth])/len(sorted_data)
        #if len(sample_list) ==1 or estL  == 0:
            # TODO: init value
        #    L = 0.01 
    else:
        print('Unkown estimation method.')
    return estL

#------------------------------------------------------------------------------------

class Base_env():
    def __init__(self, para):
        self.para = para

    def sample(self, size = None):
        pass

class Mixture_AbsGau():
    """Env for Absolute Gaussian Distribution with outliers.
    f(t) = p AbsGaux|mu1, sigma1) + (1-p) AbsGau(x|mu2, sigma2) with mu1 < mu2
    To have a IHR distribution, p needs to be small.
    When p = 1, then it's AbsGau with mu1 and sigma1. 
    """
    def __init__(self, mu1, sigma1, mu2, sigma2, p):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.p = p

    def pdf(self, x, mu, sigma):
        return 1.0/np.sqrt(2 * np.pi * sigma ** 2) * (np.exp(- 1.0/(2 * sigma**2) * (x - mu)** 2) + np.exp(- 1.0/(2 * sigma**2) * (x + mu)** 2 ))

    def cdf(self, x, mu, sigma):
        return 1.0/2 * (erf((x-mu)/ np.sqrt(2 * sigma ** 2)) + erf((x+ mu)/ np.sqrt(2 * sigma ** 2)))

    def mix_pdf(self, x):
        return (self.p * self.pdf(x, self.mu1, self.sigma1) + (1-self.p)*self.pdf(x, self.mu2,self.sigma2))

    def mix_cdf(self, x):
        return (self.p * self.cdf(x, self.mu1, self.sigma1) + (1-self.p)*self.cdf(x, self.mu2,self.sigma2))

    def hazard_rate(self, x):
        return self.mix_pdf(x)/(1 - self.mix_cdf(x))

    def sample(self, size = None):
        if size == None:
            if np.random.uniform() <= self.p:
                return np.abs(np.random.normal(self.mu1, self.sigma1)) 
            else:
                return np.abs(np.random.normal(self.mu2, self.sigma2)) 
        else:
            samples = []
            for i in range(size):
                if np.random.uniform() <= self.p:
                    s = np.abs(np.random.normal(self.mu1, self.sigma1)) 
                else:
                    s = np.abs(np.random.normal(self.mu2, self.sigma2))  
                samples.append(s)
            return np.asarray(samples)
                  
class Exp(Base_env):
    """Env for Exponential Distribution.
    """
    def __init__(self,para):
        super().__init__(para)

    def pdf(self, x):
        return self.para * np.exp(- self.para * x)

    def cdf(self,x):
        return 1 - np.exp(- self.para * x)

    def hazard_rate(self, x):
        return self.pdf(x)/(1 - self.cdf(x))

    def sample(self, size = None):
        return np.random.exponential(1.0/self.para, size)