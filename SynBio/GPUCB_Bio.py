import numpy as np
from matplotlib import pylab as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.kernel_approximation
from sklearn.gaussian_process.kernels import DotProduct
import scipy.optimize as opt

class GPUCB():
    """
    Perform GPUCB algorithm 
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6138914) 
    for Synthetic Biolody task. See the project repository 
    (https://github.com/chengsoonong/eheye/tree/master/SynBio) for details. 

    Policy:
    x_t= argmax mu_{t-1}(x)+ beta_t^{1 / 2} \sigma_{t-1}(x), 
    where beta_t = 2 log (|D| t^2 \pi^2 / 6 \delta), \delta \in (0,1)
    ------------------------------------------------------------------
    
    Attributes
    ------------------------------------------------------------------
    env: instance for Environment class, 
         having sample() method to return samples for each arm
    delta: float, 
        belong to (0,1), parameters for beta, 
        where beta is the constant to balance the first and second term for policy
    mu, sigma: array 
        mean and covariance vector for guassian process
    sample_rewards: defaultdict(list)
        key: arm index, value: list of samples for the arm
    cumu_regret: cumulative regret
    selectedActions: list
        seuquence of pulled arm index
    regret_list: list for regret for each epoch 
    X, T: list of observed samples 
    """

    def __init__(self, env, arms, arms_encoding, num_rounds,
                 init_idx, delta = 0.5):
        """
        Arguments
        ---------------------------------
        arms: list
            list of all arms (strings of len 6), e.g. 'AACTGC'
            sorted by alphabet order
        arms_encoding: list
            list of all arms one hot encodings (string of len 4 * 6),
            e.g. '100010000100000100100100'
            same order of 'arms'
        num_rounds: int
            number of rounds
        init_idx: list
            list of initial idx of arm (using the order of 'arms')
        """

        self.env = env
        self.arms = arms
        self.arms_encoding = arms_encoding
        self.num_rounds = num_rounds
        self.init_idx = init_idx
        self.delta = delta
        self.beta = 1
        
        self.label = [self.env.sample_withoutnoise(arm) for arm in arms]
        self.bestarm_idx = np.argmax(self.label)
        self.X = []
        self.T = []
        self.cumu_regret = 0
        self.regret_list = []
        self.sample_idx = []

        
        self.mu = np.zeros_like(len(self.arms))
        self.sigma = 0.5 * np.ones_like(len(self.arms))
        # self.gp = GaussianProcessRegressor(alpha = self.alpha)
        self.gp = GaussianProcessRegressor(kernel= DotProduct())
        
    def to_list(self, arms):
        """encoding arms: from strings to one-hot encoding
        
        arm: str or list of str
            string for corresponding one hot encoding for rbs1 and rbs2
            e.g. '001000100010001000100010'
            
        Return: list or array
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
        """decoding arms: from one-hot encoding to strings
        
        code: list of one hot encoding
        
        Return: string
        """
        return ''.join(str(e) for e in code)   

    def argmax_ucb(self):
        """compute upper bound.
        """
        return np.argmax(self.mu + self.sigma * self.beta)

    def learn(self, epoch):
        """sample index with maximum upper bound and update gaussian process parameters
        """
        if len(self.X) > 0 and epoch > 0:   
            self.beta = 2.0 * np.log(len(self.X) * epoch ** 2 * np.pi ** 2/ (6 * self.delta))
        
        self.gp.fit(self.X, self.T)
        self.mu, self.sigma = self.gp.predict(self.arms_encoding, return_std=True)

        idx = self.argmax_ucb()
        self.sample(idx)
        self.regret(epoch)
        
    def play(self):
        """simulate n round games
        """
        self.init_reward(self.init_idx)
        for i in range(self.num_rounds):
            self.learn(i) 
            #if i % 1000 == 0:
            #    self.plot()

    def init_reward(self, init_idx):
        
        self.arms_encoding = self.to_list(self.arms_encoding)
        self.init_arms = self.arms_encoding[init_idx]
        for i, idx in enumerate(init_idx):
            self.X.append(self.init_arms[i])
            self.sample_idx.append(idx)
            self.T.append(self.env.sample(self.arms[idx]))

    def sample(self, idx):
        """sample idx according to the gound truth
        """
        
        self.X.append(self.arms_encoding[idx])
        self.sample_idx.append(idx)
        #print(self.X)
        self.T.append(self.env.sample(self.arms[idx]))
        
    def regret(self, t):
        if self.sample_idx[-1] != self.bestarm_idx:
            #print('In ', t, 'iteration, selected: ', self.arms[self.sample_idx[-1]], ' best: ', self.arms[self.bestarm_idx])
            self.cumu_regret += self.env.sample(self.arms[self.bestarm_idx]) - self.T[-1]
        self.regret_list.append(self.cumu_regret)

    def plot(self):
        
        #fig = plt.figure()
        ax = plt.axes()
        init_len = len(self.init_arms)

        #ax.scatter(self.mu, self.label, alpha=0.5, color='g', label = 'predict')
        #ax.fill_between(test_range, preds - pred_var, preds + pred_var, facecolor='k', alpha=0.2)

        ax.plot(range(len(self.mu)), self.mu, alpha=0.5, color='g', label = 'predict')
        ax.fill_between(range(len(self.mu)), self.mu + self.sigma, self.mu - self.sigma, facecolor='k', alpha=0.2)
        #init_len = len(self.x)
        ax.scatter(self.sample_idx[:init_len], self.T[:init_len], c='b', marker='o', alpha=1.0, label = 'init sample')
        ax.scatter(self.sample_idx[init_len:], self.T[init_len:], c='r', marker='o', alpha=1.0, label = 'selected sample')
        plt.legend()
        plt.xlabel('pred')
        plt.ylabel('true')
        plt.title('GPUCB')
        plt.show()