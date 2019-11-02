from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

# Version: 25/Oct/2019
# This file implements Median-Based UCB (M-UCB) and 
# the benchmark algorithms: UCB1, UCB-V, MV-LCB, Exp3.

# Mengyan Zhang, Australian National University; Data61, CSIRO.


class Bandits_discrete(ABC):
    """Base class for bandit algorithms of a finite number of arms in discrete space.

    Arguments
    -----------------------------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    summary_stat: list
        sequence of summary statistics of arms, e.g. median, mean, etc. 
    num_rounds: int
        total number of rounds. 

    bestarm: int
        index of arm with maximum summary stat (assume only one best arm)
    sample_rewards: dict 
        keys: arm i in {0,1,2,.., K}; 
        values: list of sampled rewards for corresponding arm
    selectedActions: list
        sequence of pulled arm ID 

    ---------------For Evaluation-----------------------------------------  
    suboptimalDraws: list
        sum of draws of each sub-optimal arm 
    cumulativeRegrets: list
        Depends on the regret definition of each algorithm
        express the difference (of rewards) between drawing optimal arm 
        all the time and drawing arms based on the designed policy.  

    """
    def __init__(self, env, summary_stats, num_rounds):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        summary_stat: list
            sequence of summary statistics of arms, e.g. median, mean, etc. 
        num_rounds: int
            total number of rounds. 
        """

        self.env = env
        self.summary_stats = summary_stats
        self.num_rounds = num_rounds
        
        self.bestarm = np.argmax(summary_stats)
        self.sample_rewards = defaultdict(list)
        self.selectedActions = []
    
        self.cumulativeRegrets = []
        self.suboptimalDraws = []
        

    def init_reward(self):
        """pull each arm once and get the rewards as the initial reward 
        """
        for i, p in enumerate(self.env):
            self.sample_rewards[i].append(p.sample())
       
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
        return reward

    def evaluate(self, t):
        """Evaluate the policy
            sd: sub-optimal draws
            r: regret
    
        Parameters
        -----------------------------
        t: current time 

        Return
        -----------------------------
        None
        """
        # suboptimal draws (sd)
        sd = 0
        for key, value in self.sample_rewards.items():
            if key != self.bestarm:
                sd += len(value)
        self.suboptimalDraws.append(sd)

        # cumulative regrets (regret)
        regret = 0
        for key, value in self.sample_rewards.items():
            stats_diff = self.summary_stats[self.bestarm] - self.summary_stats[key]
            regret += stats_diff * len(value)
        self.cumulativeRegrets.append(regret)
        
class UCB_discrete(Bandits_discrete):
    """Base class of UCB algorithm. 
    """
    @abstractmethod
    def argmax_ucb(self, t):
        """Select arm index by maximise upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """

    def play(self):
        """Simulate UCB algorithms.
        """ 
        self.init_reward()
        for i in range(len(self.env),self.num_rounds):
            idx = self.argmax_ucb(i)
            self.selectedActions.append(idx)
            self.sample(idx)
            self.evaluate(i)
    
class M_UCB(UCB_discrete): 
    """M-UCB class for UCB algorithms of finite number of arms.

    Arguments
    ---------------------------------------------------------------
    est_flag: boolean 
        indicate whether estimation the lower bound of hazard rate L
        True: estimate L
        False: use the true L = f(0)/ (1 - F(0)), where f is PDF and F is CDF
    true_L_list: list
        sequence of true L (lower bound of hazard rate)
        used to compare with the results with estimated L
    hyperpara: list of  
        alpha: eps = alpha * log t,
        beta: balance empirical median and cw, 
        L_est_thr: L estimate threshold 
    """
    def __init__(self, env, summary_stats, num_rounds, **kwargs):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        summary_stat: list
            sequence of summary statistics of arms, e.g. median, mean, etc. 
        num_rounds: int
            total number of rounds. 
        """
        super().__init__(env, summary_stats, num_rounds)
        self.hyperpara = kwargs.get('hyperpara', None)
        self.est_flag = kwargs.get('est_flag', None)
        self.true_L_list = []
        self.init_L()

    def init_L(self):
        """Initialise the true_L_list for the use of true L,
        where L is the lower bound of hazard rate.
        L = f(0)/ (1 - F(0))
        """
        for i in range(len(self.env)):
            
            my_env = self.env[i]
            if hasattr(my_env, 'pdf'):
                # if pdf and cdf is defined
                x = 0
                L = my_env.pdf(x)/ (1- my_env.cdf(0))  
            else:
                L = my_env.L_estimate(self.hyperpara[-1])
            assert L > 0
            self.true_L_list.append(L)
    
    def calcu_L(self, arm_idx):
        """estimate the lower bound L of hazard rate for 
           a reward distribution specified by arm_idx
           (with the assumption of non-decreasing hazard rate)
        
        Parameter
        ---------------------------------------
        arm_idx: int
            the index of arm needed to be estimate
        
        Return
        ----------------------------------------
        L: positive float
            the lower bound of hazard rate for arm idx
        """

        if self.est_flag:
            # estimate L
            sorted_data = np.asarray(sorted(self.sample_rewards[arm_idx]))
            L = len(sorted_data[sorted_data <= self.hyperpara[-1]])/len(sorted_data)
            if L  == 0:
                L = 0.1
            return L
        else:
            # true L = f(0)/ (1 - F(0))
            return self.true_L_list[arm_idx]
    
    def argmax_ucb(self, t):
        """Select arm index by maximise upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """

        policy = []
        alpha, beta = self.hyperpara[:2]
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            emp_median = np.median(reward)
            t_i = len(reward)
            
            L = self.calcu_L(arm)
            v_t = 4.0 /( t_i * L**2)
            eps = alpha * np.log(t)
            d = np.sqrt(2 * v_t * eps) + 2 * eps * np.sqrt(v_t/t_i)
            policy.append(emp_median + beta * d)
        return np.argmax(policy)

class UCB1(UCB_discrete):
    """Implement for UCB1 algorithm

    Arguments
    ---------------------------------------------------------------
    hyperpara: list of 
        b: bound of support 
    """
    def __init__(self, env, summary_stats, num_rounds, **kwargs):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        summary_stat: list
            sequence of summary statistics of arms, e.g. median, mean, etc. 
        num_rounds: int
            total number of rounds. 
        """
        super().__init__(env, summary_stats, num_rounds)
        self.hyperpara = kwargs.get('hyperpara', None)


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
            emp_mean = np.mean(reward)
            cw = np.sqrt(2* self.hyperpara[0] ** 2 * np.log(t)/len(reward))
            ucbs.append( emp_mean + cw)
        return np.argmax(ucbs)
    
class UCB_V(UCB_discrete):
    """Implement for UCB-V algorithm
    
    Arguments
    ---------------------------------------------------------------
    hyperpara: list of 
        b: bound of support 
    """
    def __init__(self, env, summary_stats, num_rounds, **kwargs):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        summary_stat: list
            sequence of summary statistics of arms, e.g. median, mean, etc. 
        num_rounds: int
            total number of rounds. 
        """
        super().__init__(env, summary_stats, num_rounds)
        self.zeta, self.c, self.b = kwargs.get('hyperpara', None)

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
            emp_mean = np.mean(reward)
            emp_var = np.var(reward)
            eps = self.zeta * np.log(t) 
            cw = np.sqrt(2 * emp_var * eps/len(reward)) \
                + self.c * 3 * self.b * eps/len(reward)
            ucbs.append( emp_mean + cw)
        return np.argmax(ucbs)
    
class MV_LCB(UCB_discrete):
    """Implement for MV_LCB algorithm
    
    Arguments
    ---------------------------------------------------------------
    hyperpara: list of 
        rho: risk tolerance
        beta: balance the mean-variance (mv) and confidence width (cw)
    theta: term in the confidence width
    bestarm: int
        arm with minimum lower confidence width
    """
    def __init__(self, env, summary_stats, num_rounds, **kwargs):
        super().__init__(env, summary_stats, num_rounds)
        self.rho, self.beta = kwargs.get('hyperpara', None)
        self.theta = 1/(num_rounds ** 2)
        self.bestarm = np.argmin(self.summary_stats)

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
            emp_mean = np.mean(reward)
            emp_var = np.var(reward)
            MV = emp_var - self.rho * emp_mean
            cw = (5 + self.rho) * np.sqrt(np.log(1/self.theta)/len(reward))
            ucbs.append( MV - self.beta * cw)
        return np.argmin(ucbs)   
    
class Exp3(Bandits_discrete):
    """implementation Exp3 based on https://github.com/j2kun/exp3

    Arguments
    ------------------------------------
    hyperpara: list of
        gamma, an egalitarianism factor
        rewardMin, minimum value of rewards
        rewardMax, maximum value of rewards

    weights: list
        sequence of each actions' weight
    n_arm: number of arms
    """
    # 
    def __init__(self, env, summary_stats, num_rounds, **kwargs):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        summary_stat: list
            sequence of summary statistics of arms, e.g. median, mean, etc. 
        num_rounds: int
            total number of rounds. 
        
        """
        super().__init__(env, summary_stats, num_rounds)
        self.gamma, self.rewardMin, self.rewardMax = kwargs.get('hyperpara', None)

        self.n_arms = len(self.env)
        self.weights = [1.0] * self.n_arms

    def draw(self, t):
        """Pick arm index from the given list of normalized proportionally

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm 
        """
        choice = np.random.uniform(0, sum(self.weights))
        choiceIndex = 0

        for weight in self.weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

    def distr(self):
        """Normalize a list of floats to a probability distribution.
        """
        theSum = float(sum(self.weights))
        return tuple((1.0 - self.gamma) * (w / theSum) + (self.gamma / self.n_arms) for w in self.weights)

    def play(self):
        """Simulate Exp3 algorithms.
        """ 
        for i in range(self.num_rounds):
            probabilityDistribution = self.distr()
            choice = self.draw(probabilityDistribution)
            theReward = self.sample(choice)
            scaledReward = (theReward - self.rewardMin) / (self.rewardMax - self.rewardMin) # rewards scaled to 0,1

            estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
            self.weights[choice] *= np.exp(estimatedReward * self.gamma / self.n_arms) # important that we use estimated reward here!
            self.evaluate(i)