from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

import matplotlib.pyplot as plt

from codes.environment import Rewards_env

class Bandits_discrete(ABC):
    """Base class for bandit algorithms of a finite number of arms in discrete space.

    Arguments
    -----------------------------------------------------------------------------
    env: instance of Rewards_env
        Attributes: rewards_dict, labels_dict, embedded, arm_features
    num_rounds: int
        total number of rounds. 
    num_arms: int
        number of arms
    num_init: int 
        number of initialization set 
    init_list: list
        list of idx of init arms

    rewards_dict: dict of list
        rewards environment
        keys: string of embedded sequence 
            e.g. '001000100010001000100010'
        values: list of available labels
    labels_dict: dict of list
        label environment
        keys: string of embedded sequence 
            e.g. '001000100010001000100010'
        values: label (expectation of rewards list)
        sorted according to the values 
    embedded: ndarray
        num_arms x num_features
    arm_features: list
        list of arm features (string, e.g. '001000100010001000100010')
        same order as the labels dict

    bestarm_idx: int
        the arm with maximum label (assume only one arm is the best)
        TODO: consider the case for multiple best arms
    
    sample_features: list
        list of embedding features for selected arms
    sample_labels: list
        list of labels for selected arms
    sample_idx: list
        list of idx for selected arms

    ---------------For Evaluation-----------------------------------------  
    sd: int
        suboptimal draws
    r: float
        cumulative regret
    suboptimalDraws: list
        sum of draws of each sub-optimal arm 
    cumulativeRegrets: list
        Depends on the regret definition of each algorithm
        express the difference (of rewards) between drawing optimal arm 
        all the time and drawing arms based on the designed policy.  

    """
    def __init__(self, env, num_rounds, init_per, model = None):
        """
        Parameters
        ----------------------------------------------------------------------
        env: instance of Rewards_env
            Attributes: rewards_dict, labels_dict
        num_rounds: int
            total number of rounds. 
        init_per: float
            (0,1) initiation percent of arms
        model: model to fit, default is None
        """

        self.env = env
        self.num_rounds = num_rounds
        self.num_arms = len(self.env.rewards_dict)
        self.num_init = int(init_per * self.num_arms)
        self.model = model

        self.rewards_dict = self.env.rewards_dict
        self.labels_dict = self.env.labels_dict
        self.embedded = self.env.embedded
        self.arm_features = self.env.arm_features

        self.bestarm_idx = np.argmax(list(self.labels_dict.values()))
        
        self.sample_features = []
        self.sample_labels = []
        self.sample_idxs = []
    
        self.sd = 0
        self.r = 0
        self.cumulativeRegrets = []
        self.suboptimalDraws = []
    
    def to_list(self, arms):
        """From strings to list.
        
        Parameters
        ---------------------------------------------------------------
        arm: str or list of str
            string for corresponding one hot encoding for rbs1 and rbs2
            e.g. '001000100010001000100010'
            
        Returns
        ----------------------------------------------------------------
        list or array
            e.g. [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        """
        
        if type(arms) is list:
            arms_encoding = np.zeros((len(arms), len(arms[0])))
            for i,arm in enumerate(arms):
                arms_encoding[i] = np.asarray([int(e) for e in list(arm)])
            return arms_encoding.astype(int) 
        else:
            return [int(e) for e in list(arms)]
        
    def to_string(self, code):
        """From list to string.
        
        Parameters
        -------------------------------------------
        code: list
            e.g. [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        
        Returns
        -------------------------------------------
        string
            e.g. '001000100010001000100010'
        """
        return ''.join(str(int(e)) for e in code) 

    def init_reward(self):
        """initialise arms in init_list once. 
        """
        self.init_list = np.random.choice(self.num_arms, self.num_init, replace=False)
        for i in self.init_list:
            self.sample(i)
       
    def sample(self, idx):
        """sample for arm specified by idx

        Parameters
        -----------------------------
        idx: int
            the idx of arm to be sampled
        """
        self.sample_idxs.append(idx)
        features = self.arm_features[idx]
        self.sample_features.append(self.to_list(features))
        reward = self.env.sample(features)
        self.sample_labels.append(reward)

    def evaluate(self, selected_arm):
        """Evaluate the policy
            sd: sub-optimal draws
            r: regret
    
        Parameters
        -----------------------------
        selected_arm: int
            index of selected arm
        """
        if selected_arm != self.bestarm_idx:
            self.sd += 1
            bestarm_reward = self.env.sample(self.arm_features[self.bestarm_idx])
            self.r += bestarm_reward - self.sample_labels[-1]

        self.suboptimalDraws.append(self.sd)
        self.cumulativeRegrets.append(self.r)
        
class UCB_discrete(Bandits_discrete):
    """Base class of UCB algorithm. 
    """
    @abstractmethod
    def argmax_ucb(self, t):
        """Select arm index by maximise upper confidence bound.

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """

    def play(self, plot_flag = False):
        """Simulate n round games.
        """
        self.init_reward()
        for t in range(self.num_init, self.num_rounds):
            idx = self.argmax_ucb(t) 
            
            # if i % 10 == 0:
            #     self.plot()
            self.sample(idx)
            self.evaluate(idx)

class GPUCB(UCB_discrete):
    """Perform GPUCB algorithm 
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6138914) 
    for Synthetic Biolody task. See the project repository 
    (https://github.com/chengsoonong/eheye/tree/master/SynBio) for details. 

    Attributes
    --------------------------------------------------------------------
    mu: list
        posterior mean, len of number of arms.
    sigma: list
        standard deviation, len of number of arms.
    gp: instance of GaussianProcessRegressor
        TODO: other kernel methods
    """

    def __init__(self, env, num_rounds, init_per, delta = 0.5, model = None):
        """
        Parameters
        ----------------------------------------------------------------
        env: instance of Rewards_env
            Attributes: rewards_dict, labels_dict
        num_rounds: int
            total number of rounds. 
        init_per: float
            (0,1) initiation percent of arms
        delta: float
            hyperparameters for ucb.
        """

        super().__init__(env, num_rounds, init_per, model)

        self.delta = delta
        self.mu = np.zeros_like(self.num_arms)
        self.sigma = 0.5 * np.ones_like(self.num_arms)
        self.gp = self.model

    def argmax_ucb(self, t):
        """sample index with maximum upper bound and update gaussian process parameters.

           argmax mu_{t-1}(x)+ beta_t^{1 / 2} \sigma_{t-1}(x), 
           where beta_t = 2 log (|D| t^2 \pi^2 / 6 \delta), \delta \in (0,1)
        """

        if len(self.sample_features) > 0 and t > 0:   
            self.beta = 2.0 * np.log(self.num_arms * t ** 2 * np.pi ** 2/ (6 * self.delta))
        else:
            self.beta = 1

        self.gp.fit(np.asarray(self.sample_features), self.sample_labels)
        self.mu, self.sigma = self.gp.predict(np.asarray(self.to_list(self.arm_features)), return_std=True)
        

        idx = np.argmax(self.mu + self.sigma * self.beta)
        return idx
    
    def play(self, plot_flag = False, plot_per = 20):
        """Simulate n round games.

        Paramters
        ----------------------------------------------
        plot_flag: bool
             True: plot selected points during the game
        plot_per: int
            plot every plot_per rounds
        """
        self.init_reward()
        for t in range(self.num_init, self.num_rounds):
            idx = self.argmax_ucb(t) 
            
            if plot_flag:
                if t % plot_per == 0:
                    self.plot(t, plot_per)
            self.sample(idx)
            self.evaluate(idx)

    def plot(self, t, plot_per):
        """Plot for selected points during the game. 
        """
        
        #fig = plt.figure()
        ax = plt.axes()
        init_len = len(self.init_list)

        ax.plot(range(len(self.mu)), self.mu, alpha=0.5, color='g', label = 'predict')
        ax.plot(range(len(self.mu)), list(self.labels_dict.values()), alpha=0.5, color='b', label = 'true')
        ax.fill_between(range(len(self.mu)), self.mu + self.sigma, self.mu - self.sigma, facecolor='k', alpha=0.2)
        
        ax.scatter(self.sample_idxs[:init_len], self.sample_labels[:init_len], c='b', marker='o', alpha=1.0, label = 'init sample')
        # ax.scatter(self.sample_idxs[init_len:], self.sample_labels[init_len:], c='r', marker='o', alpha=1.0, label = 'selected sample')
        if init_len < t - plot_per:
            start_round = t - plot_per
        else:
            start_round = init_len
        ax.scatter(self.sample_idxs[start_round:t-1], self.sample_labels[start_round:t-1], c='r', marker='o', alpha=1.0, label = 'selected sample')
        
        plt.legend()
        plt.xlabel('Arm Index')
        plt.ylabel('Label')
        plt.title('GPUCB ' + str(t-plot_per) + '~' + str(t) + ' rounds')
        plt.show()

class Random(UCB_discrete):
    """Randomly select sequences."""

    def argmax_ucb(self, t):
        """Randomly select the sequence. 
        """
        return np.random.choice(self.num_arms)
    