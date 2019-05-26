from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
from qreg import QRegressor

class UCB_discrete(ABC):
    """Base class for UCB algorithms of finite number of arms.
    """
    def __init__(self, env, num_rounds, bestarm):
        """
        Arguments
        ------------------------------------
        env: list
            sequence of instances of Environment (distribution reward of arms)
        num_rounds: int
            total number of rounds
        bestarm: int
            idx of best arm wrt true distribution (to calculate regret)
        sample_rewards: dict 
            keys: arm ID (0,1,2,..); 
            values: list of sampled rewards for corresponding arm
        cumulativeReward: float
            cumulative reward up to current round
        bestActionCumulativeReward: float
            cumulative reward for pulling the arm 
            with highest expected value up to current round
        cumulativeRegrets: list
            difference between cumulativeReward and bestActionCumulativeReward
            for each round 
        selectedActions: list
            sequence of pulled arm ID 
        num_palyed: list
            sequence of number of the times of each arm was played
        """
        self.env = env
        self.num_rounds = num_rounds
        self.bestarm = bestarm
        self.sample_rewards = defaultdict(list)
        self.cumulativeReward = 0.0
        self.bestActionCumulativeReward = 0.0
        self.cumulativeRegrets = []
        self.selectedActions = []
        #self.num_played = []

    @abstractmethod
    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        
    #def regret(self, idx, reward):
    def regret(self):
        """Calculate cumulative regret and record it

        Parameters
        -----------------------------
        idx: int
            the idx of arm with maximum ucb in the current round
        reward: float
            sample reward for the current round

        Return
        -----------------------------
        None
        """

        """
        if idx != self.bestarm:
            self.bestActionCumulativeReward += self.env[self.bestarm].sample()
        else:
            self.bestActionCumulativeReward += reward
        CumulatveRegret = self.bestActionCumulativeReward - self.cumulativeReward
        self.cumulativeRegrets.append(CumulatveRegret)
        """
        my_regret = 0
        for key, value in self.sample_rewards.items():
            mu_diff = self.env[self.bestarm].loc - self.env[key].loc
            my_regret += mu_diff * len(value)
        self.cumulativeRegrets.append(my_regret)
    
    def init_reward(self):
        """pull each arm once and get the rewards as the inital reward 
        """
        for i, p in enumerate(self.env):
            self.sample_rewards[i].append(p.sample())
            #self.num_played.append(1)

    def play(self):
        """Simulate UCB algorithms for specified rounds.
        """
        
        self.init_reward()
        for i in range(len(self.env),self.num_rounds):
            #print('Round: ', i)
            idx = self.argmax_ucb(i)
            self.selectedActions.append(idx)
            #print('select arm: ', idx)
            reward = self.sample(idx)
            #self.num_played[idx] += 1
            #self.regret(idx, reward)
            self.regret()
        return self.cumulativeRegrets

    def sample(self, idx):
        """sample for arm specified by idx

        Parameters
        -----------------------------
        idx: int
            the idx of arm with maximum ucb in the current round
        
        Return
        ------------------------------
        reward: float
            sampled reward from idx arm
        """
        reward = self.env[idx].sample()
        self.sample_rewards[idx].append(reward)
        self.cumulativeReward += reward
        return reward

class UCB_os(UCB_discrete):
    """class for UCB of order statistics 
    (in terms of absolute value of standard gaussian distribution)
    regret is evaluted in terms of median rather than mean. 

    Arguments
    -------------------------------------------------------------
    medians: list
        sequence of medians of arms 
    """
    def __init__(self, env, num_rounds, medians):
        self.medians = medians
        bestarm = np.argmax(self.medians)
        super().__init__(env, num_rounds, bestarm)

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        policy = []
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            emp_median = np.median(reward)
            t_i = len(reward)
            v_t = 8.0/( t_i * np.log(2))
            eps = 4 * np.log(t)
            b = np.sqrt(2 * v_t * eps) + 2 * eps * np.sqrt(v_t/t_i)
            policy.append(emp_median + b)
            #print(policy)
        return np.argmax(policy)

    def regret(self):
        """Calculate cumulative regret and record it

        Parameters
        -----------------------------
        idx: int
            the idx of arm with maximum ucb in the current round
        reward: float
            sample reward for the current round

        Return
        -----------------------------
        None
        """
        my_regret = 0
        for key, value in self.sample_rewards.items():
            mu_diff = self.medians[self.bestarm] - self.medians[key]
            my_regret += mu_diff * len(value)
        self.cumulativeRegrets.append(my_regret)


class QuantUCB(UCB_discrete):
    """Base class for Quantile UCB
    """
    def __init__(self, env, num_rounds, bestarm):
        super().__init__(env, num_rounds, bestarm)

    def quantile(self, data, alpha):
        """calculate empirical alpha-quantile for given data samples.

        Parameters
        -----------------------------------
        data: list
            sequence of sample rewards
        alpha: 
            level of quantile

        Return
        ------------------------------------
        quantile: float
            alpha level quantile of given data
        """
        
        data = np.sort(data)
        idx = int(len(data) * alpha)
        return data[idx]

    def sigmoid(self, x):
        return 1/ (1+ np.exp(-x))

    def alpha_level(self, t, n_selected):
        """calculate the alpha level for quantile.
        
        Parameters
        -----------------------------------
        t: int
            the current round

        n_selected: int
            the number of times arm j was selected up to the current round

        Return
        -----------------------------------
        alpha: float 
            between 0 and 1, alpha level for quantile
        """
        #alpha = np.sqrt(np.log(t)/n_selected)
        alpha = self.sigmoid(np.log(np.sqrt(10.0/n_selected)))
        #print('alpha: ', alpha)
        return alpha

        # ugly control here, need to be fixed
        #if alpha >= 1:
        #    return 0.8
        #else: 
        #    return alpha

class QuantUCB_Gau(UCB_discrete):
    """Quantile UCB with Gaussian rewards. 
       For now, we assume the env is exposed to calculate alpha but not exposed to player.
       For now, instead of using estimated quantile, we first plug alpha into true quantile function.
    """
    def __init__(self, env, num_rounds, bestarm):
        super().__init__(env, num_rounds, bestarm)

    def emp_quantile(self, data, alpha):
        """calculate empirical alpha-quantile for given data samples.

        Parameters
        -----------------------------------
        data: list
            sequence of sample rewards
        alpha: 
            level of quantile

        Return
        ------------------------------------
        quantile: float
            alpha level quantile of given data
        """
        
        data = np.sort(data)
        idx = int(len(data) * alpha)
        return data[idx]

    def linear_inter_quant(self, alpha, data):
        """implement linear interpolation for quantile estimation.
        
        Parameters
        -----------------------------------
        data: list
            sequence of sample rewards
        alpha: 
            level of quantile

        Return
        ------------------------------------
        quantile: float
            alpha level quantile of given data
        """
        size = len(data) 
        data = list(data)
        data.append(-2)
        data = np.sort(data)
        s = int(alpha * size) 
        rate = (data[s + 1] - data[s]) * size
        #b = data[s] * (1-s) - data[s+1]
        #return rate * alpha + b
        return rate * (alpha - float(s)/size) + data[s]

    def quantile(self, i, alpha):
        """Calculate true quantile for distribution of given alpha
        """
        #return self.env[i].loc + self.env[i].scale * np.sqrt(2) * self.inv_erf(2 * alpha - 1)
        return np.mean(self.sample_rewards[i]) + self.env[i].scale * np.sqrt(2) * self.inv_erf(2 * alpha - 1)
    
    def erf(self,x):
        """Approximate erf with maximum error 1.5 * 10^(-7)
        """
        a1=  0.0705230784
        a2=  0.0422820123
        a3=  0.0092705272
        a4=  0.0001520143
        a5=  0.0002765672
        a6=  0.0000430638

        return 1.0 - 1.0/(1+ a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5 + a6 * x**6)**16

    def inv_erf(self,x):
        "Approximate inverse erf"
        a = 0.140012
        temp = 2.0/(np.pi * a) + np.log(1-x**2)/2.0
        return np.sign(x) * np.sqrt((np.sqrt(temp**2 - np.log(1-x**2)/a) - temp))

    def alpha_level(self, t, n_selected):
        """calculate the alpha level for quantile.
        
        Parameters
        -----------------------------------
        t: int
            the current round

        n_selected: int
            the number of times arm j was selected up to the current round

        Return
        -----------------------------------
        alpha: float 
            between 0 and 1, alpha level for quantile
        """
        return 0.5 * (self.erf(2 * np.sqrt(np.log(t)/n_selected)) + 1)

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        ucbs = []
        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            mean_reward = np.mean(reward)
            
            # choice 1: compute ucb using true quantiles
            # quant = self.quantile(arm, self.alpha_level(t, len(reward)))
            
            # choice 2: compute ucb using estimated quantiles with linear interpolation
            # quant = self.linear_inter_quant(self.alpha_level(t, len(reward)), reward) 

            # choice 3: empirical quantiles + sqrt(2lnt/s)
            # quant = self.linear_inter_quant(self.alpha_level(t, len(reward)), reward) + np.sqrt(2 * np.log(t)/len(reward))

            # choice 4: empirical quantile difference with fixed alpha (0.9, 0.1) + sqrt(2 lnt/s)
            # quant = self.linear_inter_quant(0.9, reward) - self.linear_inter_quant(0.5, reward) + np.sqrt(2 * np.log(t)/len(reward))

            # choice 5: ucb1 tuned
            quant = np.var(reward) + np.sqrt(2 * np.log(t)/len(reward))

            #print('arm ', arm, 'mean reward: ', mean_reward, ' quant:', quant)
            #print('mean + quant: ', mean_reward + quant)
            
            # mean + beta * quantile
            beta = np.sqrt(np.log(self.num_rounds)/len(reward))

            # mean + quantile 
            #beta = 1

            ucbs.append(mean_reward + beta * quant)
            
        assert len(ucbs) == len(self.env)
        return np.argmax(ucbs)

class QuantUCB_MUQ(QuantUCB):
    """QuantUCB by maximizing upper quantiles
    """
    def __init__(self, env, num_rounds, bestarm):
        super().__init__(env, num_rounds, bestarm)

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        ucbs = []
        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            quant = self.quantile(reward, self.alpha_level(t, len(reward)))
            #print('arm ', arm, 'mean reward: ', np.mean(reward), ' quant:', quant)
            #print('np.sqrt(np.log(t)) * quant: ', np.sqrt(np.log(t)/len(reward)) * quant)
            ucbs.append(np.mean(reward) + np.sqrt(np.log(t)/len(reward)) * quant)
            
        assert len(ucbs) == len(self.env)
        return np.argmax(ucbs)

class QuantUCB_MMQD(QuantUCB):
    """QuantUCB by maximizing the sum of mean and quantile differences
    """
    def __init__(self, env, num_rounds, bestarm):
        super().__init__(env, num_rounds, bestarm)

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        ucbs = []
        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            quant = self.quantile(reward, self.alpha_level(t, len(reward)))
            median = self.quantile(reward, 0.5)
            #print('arm ', arm, 'mean reward: ', np.mean(reward), ' quant:', quant, ' median: ', median)
            ucbs.append(np.mean(reward) + quant - median)
            
        assert len(ucbs) == len(self.env)
        return np.argmax(ucbs)

class CVaRUCB(UCB_discrete):
    """Base class for CVaR UCB
    """
    def CVaR(self, data, alpha):
        """calculate alpha-CVaR for given data samples.

            Parameters
            -----------------------------------
            data: list
                sequence of sample rewards
            alpha: 
                level of quantile

            Return
            ------------------------------------
            quantile: float
                alpha level quantile of given data
        """
        data = np.sort(data)
        idx = int(len(data) * alpha)
        CVaR = np.mean(data[idx:])
        return CVaR

    def sigmoid(self, x):
        return 1/ (1+ np.exp(-x))

    def alpha_level(self, t, n_selected):
        """calculate the alpha level for quantile.
        
        Parameters
        -----------------------------------
        t: int
            the current round

        n_selected: int
            the number of times arm j was selected up to the current round

        Return
        -----------------------------------
        alpha: float 
            between 0 and 1, alpha level for quantile
        """
        alpha = np.sqrt(np.log(t)/n_selected)
        return self.sigmoid(alpha)
        ##print(alpha)

        # ugly control here, need to be fixed
        #if alpha >= 1:
        #    return 0.8
        #else: 
        #    return alpha
    
    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        ucbs = []
        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            CVaR = self.CVaR(reward, self.alpha_level(t, len(reward)))
            ucbs.append(np.mean(reward) + CVaR)
        assert len(ucbs) == len(self.env)
        return np.argmax(ucbs)


class UCB1(UCB_discrete):
    """Implement for UCB1 algorithm
    """
    def __init__(self, env, num_rounds, bestarm):
        super().__init__(env, num_rounds, bestarm)

    def argmax_ucb(self, t):
        """
        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        ucbs = []
        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            ucbs.append(np.mean(reward) + np.sqrt(2*np.log(t)/len(reward)))
        assert len(ucbs) == len(self.env)
        return np.argmax(ucbs)

class Environment():
    """Environment for distribution reward of arms.
    """
    def __init__(self, loc=0.0, scale=1.0, skewness = 0.0):
        self.loc = loc
        self.scale = scale
        self.skewness = skewness

    """
    def sample(self):
        uniformly generate x (between 0 and 1), 
           generate normal or skewed normal samples
        
        sigma = self.skewness/np.sqrt(1.0 + self.skewness**2) 
        u0,v = np.random.randn(2)
        u1 = sigma*u0 + np.sqrt(1.0-sigma**2) * v 
        if u0 >= 0:
            return u1*self.scale + self.loc
        return (-u1)*self.scale + self.loc 
    """

    def sample(self):
        return np.abs(np.random.normal(self.loc, self.scale))

