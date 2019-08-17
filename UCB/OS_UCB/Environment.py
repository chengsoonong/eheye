import numpy as np
from collections import defaultdict
from scipy.special import erf


class Base_env():
    def __init__(self, para):
        self.para = para

    def sample(self, size = None):
        pass

class Clinical_env():
    def __init__(self, data):
        """
        Parameters:
        ------------------------------------------
        num_arms: int
        data: sequence of samples 
        """

        self.data = data
    
    def sample(self):
        return np.random.choice(self.data)

    def L_estimate(self):
        sorted_data = np.asarray(sorted(self.data))
        L = len(sorted_data[sorted_data <= 100])/len(sorted_data)
        return L
        
class AbsGau(Base_env):
    def __init__(self, para):
        super().__init__(para) 

    def pdf(self, x):
        return 2.0/(self.para * np.sqrt(2 * np.pi)) * np.exp(- x ** 2/ (2 * self.para)) 

    def cdf(self, x):
        return erf(x/ (self.para * np.sqrt(2)))

    def sample(self, size = None):
        return np.abs(np.random.normal(0, self.para, size))

class AbsGau_Outlier(Base_env):
    def __init__(self, para):
        super().__init__(para) 

    def pdf(self, x):
        return 2.0/(self.para * np.sqrt(2 * np.pi)) * np.exp(- x ** 2/ (2 * self.para)) 

    def cdf(self, x):
        return erf(x/ (self.para * np.sqrt(2)))

    def sample(self, size = None):
        if size == None:
            if np.random.uniform() <= 0.95:
                return np.abs(np.random.normal(0, self.para, size)) 
            else:
                return np.abs(np.random.normal(50,0.1)) 
        else:
            samples = []
            for i in range(size):
                if np.random.uniform() <= 0.95:
                    s = np.abs(np.random.normal(0, self.para)) 
                else:
                    s = np.abs(np.random.normal(50,0.1)) 
                samples.append(s)
            return np.asarray(samples)
                

class Exp(Base_env):
    def __init__(self,para):
        super().__init__(para)

    def pdf(self, x):
        return self.para * np.exp(- self.para * x)

    def cdf(self,x):
        return 1 - np.exp(- self.para * x)

    def sample(self, size = None):
        return np.random.exponential(1.0/self.para, size)

class Exp_Outlier(Base_env):
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
                return np.abs(np.random.normal(50,0.1)) 
        else:
            samples = []
            for i in range(size):
                if np.random.uniform() <= 0.95:
                    s = np.random.exponential(1.0/self.para) 
                else:
                    s = np.abs(np.random.normal(50,0.1)) 
                samples.append(s)
            return np.asarray(samples)


class Comb(Base_env):
    def __init__(self,para):
        super().__init__(para)
   
    def sample(self, arm, size = None):
        """
        assum the first arm is AbsGau, the other two are Exp
        """
        if arm == 0:
            return np.abs(np.random.normal(0, self.para, size))
        else:
            return np.random.exponential(self.para, size)

def setup_env(num_arms, envs_setting):
    """
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

    Return:
    --------------------------------------------------
    results_env: dict of list
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of env instances
    medians: dict of list
        keys: name of environment 
            e.g. AbsGau_Outlier_[0.5, 1, 1.5]
        values: list of medians
    
    """
    
    rewards_env = defaultdict(list)
    medians = defaultdict(list)
    num_samples = 10000

    '''
    for env, para in envs_setting.items():
        
        for i in range(num_arms):
            current_env = env(para[i])
            rewards_env[env_name].append(current_env)
            medians[env_name].append(np.median(current_env.sample(i, num_samples)))
    '''

    for envs_dict in envs_setting:
        
        name = ''
        for env, para_list in envs_dict.items():
            env_name = str(env).split('.')[-1][:-2]
            name += env_name + '_' + str(para_list)

        for env, para_list in envs_dict.items():
            for para in para_list:
                current_env = env(para)
                rewards_env[name].append(current_env)
                medians[name].append(np.median(current_env.sample(num_samples)))
    return rewards_env, medians



