import numpy as np
from collections import defaultdict


class Base_env():
    def __init__(self, para):
        self.para = para

    def sample(self, arm, size = None):
        pass
  
class AbsGau(Base_env):
    def __init__(self, para):
        super().__init__(para) 

    def sample(self, arm, size = None):
        return np.abs(np.random.normal(0, self.para, size))

class Exp(Base_env):
    def __init__(self,para):
        super().__init__(para)

    def sample(self, arm, size = None):
        return np.random.exponential(self.para, size)

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
    rewards_env = defaultdict(list)
    medians = defaultdict(list)
    num_samples = 10000

    for env, para in envs_setting.items():
        env_name = str(env).split('.')[-1][:-2]
        for i in range(num_arms):
            current_env = env(para[i])
            rewards_env[env_name].append(current_env)
            medians[env_name].append(np.median(current_env.sample(i, num_samples)))

    return rewards_env, medians



