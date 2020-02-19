import numpy as np
from collections import defaultdict
from scipy.special import erf

# Version: Feb/2020
# This file implements the environment construction for bandits algorithm. 
# The environments include: 
#   Simulated environments:
#       Light tailed distributions: absolute Gaussian, Exponential
#       Heavy tailed distributions: log normal 
#       Outliers include two types: 
#           distributional outlier (centered absolute Gaussian); 
#           arbitrary outlier (uniformly sampled from random set).
#   Clinical environment: sample from data

OUTLIER_CENTER = 20

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

class AbsGau_Arb_Outlier(Base_env):
    """Env for Absolute Gaussian Distribution with arbitrary outliers.
    """
    def __init__(self, para, random_set):
        super().__init__(para) 
        self.random_set = random_set

    def pdf(self, x):
        return 2.0/(self.para * np.sqrt(2 * np.pi)) * np.exp(- x ** 2/ (2 * self.para)) 

    def cdf(self, x):
        return erf(x/ (self.para * np.sqrt(2)))

    def sample(self, size = None):
        if size == None:
            if np.random.uniform() <= 0.95:
                return np.abs(np.random.normal(0, self.para, size)) 
            else:
                return np.random.choice(self.random_set)
        else:
            samples = []
            for i in range(size):
                if np.random.uniform() <= 0.95:
                    s = np.abs(np.random.normal(0, self.para)) 
                else:
                    s = np.random.choice(self.random_set)
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

    def sample(self, size = None):
        return np.random.exponential(1.0/self.para, size)

class Exp_Outlier(Base_env):
    """Env for Exponential Distribution with outliers.
    """
    def __init__(self, para):
        super().__init__(para) 

    def pdf(self, x):
        return self.para * np.exp(- self.para * x)

    def cdf(self,x):
        return 1 - np.exp(- self.para * x)

    def sample(self, size = None):
        if size == None:
            if np.random.uniform() <= 0.95:
                return np.random.exponential(1.0/self.para, size)
            else:
                return np.abs(np.random.normal(OUTLIER_CENTER,0.1)) 
        else:
            samples = []
            for i in range(size):
                if np.random.uniform() <= 0.95:
                    s = np.random.exponential(1.0/self.para) 
                else:
                    s = np.abs(np.random.normal(OUTLIER_CENTER,0.1)) 
                samples.append(s)
            return np.asarray(samples)

class Exp_Arb_Outlier(Base_env):
    """Env for Exponential Distribution with arbitrary outliers.
    """
    def __init__(self, para, random_set):
        super().__init__(para)
        self.random_set = random_set 

    def pdf(self, x):
        return self.para * np.exp(- self.para * x)

    def cdf(self,x):
        return 1 - np.exp(- self.para * x)

    def sample(self, size = None):
        if size == None:
            if np.random.uniform() <= 0.95:
                return np.random.exponential(1.0/self.para, size)
            else:
                return np.random.choice(self.random_set)
        else:
            samples = []
            for i in range(size):
                if np.random.uniform() <= 0.95:
                    s = np.random.exponential(1.0/self.para) 
                else:
                    s = np.random.choice(self.random_set)
                samples.append(s)
            return np.asarray(samples)

# ---------------------------------------------------------------------------
# Heavy tail environments (For comparison)

from scipy.stats import pareto

class Weibull():
    def __init__(self, shape):
        """
        shape: float
            shape of the distribution. should be greater than zero

        scale is default as 1. 
        """
        self.shape = shape

    def pdf(self, x):
        if x == 0:
            return 0
        return self.shape * (x ** (self.shape - 1)) * np.exp((- x) ** self.shape)

    def cdf(self,x):
        if x == 0:
            return 0
        return 1 - np.exp((- x) ** self.shape)

    def sample(self, size = None):
        return np.random.weibull(self.shape, size)

# -----------------------------------------------------------------------------------
# Clinical environment

class Clinical_env():
    """Environment class for clinical data.
    """
    def __init__(self, data):
        """
        Parameters:
        ------------------------------------------
        data: sequence of samples 
        """
        self.data = data
    
    def sample(self):
        return np.random.choice(self.data)

    def L_estimate(self, thr):
        """Estimation of lower bound of hazard rate (L).

        Parameters
        ------------------------------------------------
        thr: float
            threshold of estimation.
        """
        sorted_data = np.asarray(sorted(self.data))
        L = len(sorted_data[sorted_data <= thr])/len(sorted_data)
        return L

# ---------------------------------------------------------------------------------
def setup_env(num_arms, envs_setting, random_set = None):
    """Setup environment for simulations.

    Parameter:
    --------------------------------------------------
    num_arms: int
        number of arms
    envs_setting: list of env dict
        keys: instance of environment class 
            (AbsGau, Exp, AbsGau_Outlier, Exp_Outlier)
        values: list of parameters
        e.g.environments = [
                {AbsGau: [0.5, 1.0, 1.5]}, 
                {Exp:    [2.0, 1.0, 1.5]},
                {AbsGau: [0.5], Exp: [1.0, 1.5]}
               ]

    Return:
    --------------------------------------------------
    results_env: dict of list
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of env instances
    medians: dict of list for medians
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of medians
    means: dict of list for means
    mvs: dict of list for mean-variance
    samples: dict of list for samples
    """
    
    rewards_env = defaultdict(list)
    medians = defaultdict(list)
    samples = defaultdict(list)
    means = defaultdict(list)
    mvs = defaultdict(list)
    num_samples = 10000

    for envs_dict in envs_setting:
        name = ''
        for env, para_list in envs_dict.items():
            env_name = str(env).split('.')[-1][:-2]
            name += env_name + '_' + str(para_list)

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
                medians[name].append(np.median(sample))
                means[name].append(np.mean(sample))
                # mvs[name].append(np.var(sample) - paras[0] * np.mean(sample))
    return rewards_env, medians, means, mvs, samples