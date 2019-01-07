import numpy as np
from matplotlib import pylab as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.kernel_approximation
import scipy.optimize as opt
from quantregression import QuantReg

class GPUCB(object):
    """Perform GPUCB algorithm on environment
    
    Attributes
    --------------------------------
    
    environment: instance of DummyEnvironment, generate samples using x
    x, t: list of all index and values of samples
    maxlabel: max label of t
    beta: parameter for gaussian process
        beta can be fixed as specified by arguments
        beta also can be changed along with epoches
    alpha, mu, sigma: parameter for guassian process
    X, T: list of observed samples 
    cumu_regret: cumulative regret
    regret_list: list for regret for each epoch 
    """

    def __init__(self, x, t, alpha, beta=1.):
        """
        Arguments
        ---------------------------------
        x
            list of all index of samples
        t
            list of all values of samples
        alpha
            parameter for gaussian process 
        beta
            parameter for upper bound
        """

        #self.environment = environment
        self.x = x
        self.t = t
        self.maxlabel = np.max(self.t)
        self.alpha = alpha
        self.beta = beta
        self.mu = np.zeros_like(x)
        self.sigma = 0.5 * np.ones_like(x)
        self.X = []
        self.T = []
        self.cumu_regret = 0
        self.regret_list = []
        self.gp = GaussianProcessRegressor(alpha = self.alpha)

    def argmax_ucb(self):
        """compute upper bound.
        """
        return np.argmax(self.mu + self.sigma * self.beta)

    def regret(self):
        """Compute regret and cumulative regret   
        """
        self.cumu_regret += self.maxlabel - self.T[-1]
        self.regret_list.append(self.cumu_regret)

    def learn(self, epoch):
        """sample index with maximum upper bound and update gaussian process parameters
        """
        if len(self.X) > 0:   
            self.beta = 2.0 * np.log(len(self.X) * (epoch + 1.0))/20
        idx = self.argmax_ucb()
        self.sample(idx)
        self.regret()
        self.gp.fit(self.X, self.T)
        self.mu, self.sigma = self.gp.predict(self.x.reshape((self.x.shape[0],1)), return_std=True)

    def sample(self, idx):
        """sample idx according to the gound truth
        """
        self.X.append([self.x[idx]])
        self.T.append(self.t[idx])

    def plot(self):
        fig = plt.figure()
        ax = plt.axes()

        min_val = min(self.x)
        max_val = max(self.x)
        test_range = np.arange(min_val - 1, max_val + 1, 0.1)
        num_test = len(test_range)
        #test_range.shape = (num_test, 1)

        (preds, pred_var) = self.gp.predict(test_range.reshape(num_test,1), return_std= True)
        ax.plot(test_range, preds, alpha=0.5, color='g', label = 'predict')
        ax.fill_between(test_range, preds - pred_var, preds + pred_var, facecolor='k', alpha=0.2)

        #ax.plot(self.x, self.mu, alpha=0.5, color='g', label = 'predict')
        #ax.fill_between(self.x, self.mu + self.sigma, self.mu - self.sigma, facecolor='k', alpha=0.2)
        ax.scatter(self.X, self.T, c='r', marker='o', alpha=1.0, label = 'groundtruth')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('GPUCB')
        plt.show()

class QuantUCB(object):
    """Perform QuantUCB algorithm on environment
    
    Attributes
    --------------------------------
    
    environment: instance of DummyEnvironment, generate samples using x
    x, t: list of all index and values of samples
    maxlabel: max label of t

    beta: parameter for adjust predictions and variance 
        beta can be fixed as specified by arguments
        beta also can be changed along with epoches
    uq, lq: upper and lower quantile
    uq_rate: adjust upper quantile during updating 

    predict, ub, lb: prediction median, 
        upper and lower bound based on the specidied uq and lq

    D: kernel dimensions
    sampler: kernel sampler

    X, T: list of observed samples 
    cumu_regret: cumulative regret
    regret_list: list for regret for each epoch 
    """

    def __init__(self, x, t, max_method='MUB',
                 uq=0.9, lq=0.1, uq_rate=0.0, beta=1.):
        
        #self.environment = environment
        #self.x = np.stack([x.ravel(), np.ones_like(x.ravel())]).T
        self.x = x
        self.t = t
        self.maxlabel = np.max(self.t)

        self.max_method = max_method
        self.beta = beta
        self.uq = uq
        self.lq = lq
        self.uq_rate = uq_rate

        self.X = []
        self.T = []
        self.cumu_regret = 0
        self.regret_list = []
        
        self.predict = np.zeros_like(self.x)
        self.ub = 0.5 * np.ones_like(self.x)
        self.lb = -0.5 * np.ones_like(self.x)
        
        self.D = 50
        self.QuantReg = QuantReg(self.D)
        #self.sampler = sklearn.kernel_approximation.RBFSampler(n_components= self.D, gamma=0.1)
        
    def argmax_ucb(self, max_method, epoch):
        if max_method == 'MUB':
            # use upper bound
            if self.uq < 0.95:
                self.uq += self.uq_rate
            return np.argmax(self.ub)
        elif max_method == 'MPV':
            # use predict + uncertainty
            if len(self.X) > 0:   
                self.beta = np.log(len(self.X) * (epoch + 1.0))/20
            return np.argmax(self.predict + 0.5 * abs(self.ub - self.lb) * self.beta)
        else:
            raise ValueError

    def regret(self):
        self.cumu_regret += self.maxlabel - self.T[-1]
        self.regret_list.append(self.cumu_regret)
        
    def learn(self, epoch):
        idx = self.argmax_ucb(self.max_method, epoch)
        self.sample(idx)
        self.regret()

        #print(self.X)
        #print(self.T)
        self.QuantReg.fit(self.X, self.T, self.uq, self.lq)
        self.predict, self.ub, self.lb = self.QuantReg.predict(self.x)

    def sample(self, idx):
        self.X.append(self.x[idx])
        self.T.append(self.t[idx])
   
    def plot(self):
        fig = plt.figure()
        ax = plt.axes()

        min_val = min(self.x)
        max_val = max(self.x)
        test_range = np.arange(min_val - 1, max_val + 1, 0.1)
        num_test = len(test_range)
        #test_range.shape = (num_test, 1)

        #test_range = self.x
        preds, ub, lb = self.QuantReg.predict(test_range)
        
        ax.plot(test_range, preds, alpha=0.5, color='g', label = 'predict')
        ax.fill_between(test_range, lb, ub, facecolor='k', alpha=0.2)
        ax.scatter(self.X, self.T, c='r', marker='o', alpha=1.0, label = 'sample')
        #plt.savefig('fig_%02d.png' % len(self.X))
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('QuantUCB')
        plt.show()

class DummyEnvironment(object):
    """Environment to generate samples with noise.
    """
    def sin_sample(self, x):
        """Generating model is sin(x). 
        Noise model is normal distribution.
        """
        noise = np.random.normal(0,0.1,1)
        environ_gene = np.sin(x)

        return np.sin(x) + noise[0]
        #return np.sin(x)

