from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

class UCB_discrete(ABC):
    """Base class for UCB algorithms of finite number of arms.
    """
    def __init__(self, env, num_rounds, bestarm):
        """
        Arguments
        ------------------------------------
        env: list
            sequence of distribution reward of arms
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
        
    def regret(self, idx, reward):
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
        if idx != self.bestarm:
            self.bestActionCumulativeReward += self.env[self.bestarm].sample()
        else:
            self.bestActionCumulativeReward += reward
        CumulatveRegret = self.bestActionCumulativeReward - self.cumulativeReward
        self.cumulativeRegrets.append(CumulatveRegret)
        
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
            print('Round: ', i)
            idx = self.argmax_ucb(i)
            print('select arm: ', idx)
            reward = self.sample(idx)
            #self.num_played[idx] += 1
            self.regret(idx, reward)
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

class QuantUCB(UCB_discrete):
    """Base class for Quantile UCB
    """
    def __init__(self, env, num_rounds, bestarm):
        super().__init__(env, num_rounds, bestarm)

    def quantile(self, data, alpha):
        """calculate alpha-quantile for given data samples.

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
        idx = int(len(data) * alpha) - 1
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
        print('alpha: ', alpha)
        return alpha

        # ugly control here, need to be fixed
        #if alpha >= 1:
        #    return 0.8
        #else: 
        #    return alpha

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
            print('arm ', arm, 'mean reward: ', np.mean(reward), ' quant:', quant)
            print('np.sqrt(np.log(t)) * quant: ', np.sqrt(np.log(t)/len(reward)) * quant)
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
            print('arm ', arm, 'mean reward: ', np.mean(reward), ' quant:', quant, ' median: ', median)
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
        #print(alpha)

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

    def sample(self):
        sigma = self.skewness/np.sqrt(1.0 + self.skewness**2) 
        u0,v = np.random.randn(2)
        u1 = sigma*u0 + np.sqrt(1.0-sigma**2) * v 
        if u0 >= 0:
            return u1*self.scale + self.loc
        return (-u1)*self.scale + self.loc 