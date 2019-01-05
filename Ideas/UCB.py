import numpy as np
from matplotlib import pylab as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.kernel_approximation
import scipy.optimize as opt

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
            print(epoch)
            print(self.beta)
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
        print(preds.shape)
        print(pred_var.shape)
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

    def __init__(self, x, t, uq, lq, uq_rate=0.001, beta=1.):
        
        #self.environment = environment
        self.x = np.stack([x.ravel(), np.ones_like(x.ravel())]).T
        self.t = t
        self.maxlabel = np.max(self.t)

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
        self.sampler = sklearn.kernel_approximation.RBFSampler(n_components= self.D, gamma=0.1)
        
    def argmax_ucb(self):
        # use upper bound
        return np.argmax(self.ub)
        
        # use predict + uncertainty
        # return np.argmax(self.predict + 0.5 * abs(self.ub - self.lb) * np.sqrt(self.beta))

    def regret(self):
        self.cumu_regret += self.maxlabel - self.T[-1]
        self.regret_list.append(self.cumu_regret)
        
    def cost(self, W, X, Y, q):
        predictions = X @ W
        return np.where(
            predictions > Y,
            (1 - q) * np.abs(Y - predictions),
            q * np.abs(Y - predictions)).sum()
    
    def opti(self,X,Y):
        self.opt_w = opt.fmin_bfgs(self.cost, np.zeros((self.D,)), args=(X, Y, 0.5))
        self.opt_u = opt.fmin_bfgs(self.cost, np.zeros((self.D,)), args=(X, Y, self.uq), disp = False)
        self.opt_l = opt.fmin_bfgs(self.cost, np.zeros((self.D,)), args=(X, Y, self.lq), disp = False)

    def learn(self):
        idx = self.argmax_ucb()
        self.sample(idx)
        self.regret()
        
        X_ = np.asarray(self.X)
        X_ = np.stack([X_.ravel(), np.ones_like(X_.ravel())]).T
        #print('X_ shape:', X_.shape)
        #print('X_', X_)
        
        X_rbf = self.sampler.fit_transform(X_)
        #print('X_rbf shape:', X_rbf.shape)
        #print('X_rbf', X_rbf)
        if self.uq < 0.95:
            self.uq += self.uq_rate
        self.opti(X_rbf, np.asarray(self.T))
        
        self.x_ = self.sampler.transform(self.x)
        #print('self.x_ shape:', self.x_.shape)
        #print('self.x_', self.x_)
        self.predict = self.x_ @ self.opt_w
        #print('predictions:', self.predict)
        self.ub = self.x_ @ self.opt_u
        #print('upper bound:', self.ub)
        self.lb = self.x_ @ self.opt_l
        #print('lower bound:', self.lb)

    def sample(self, idx):
        self.X.append(self.x[idx,0])
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
        print(preds.shape)
        print(pred_var.shape)
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
   
    def plot(self):
        fig = plt.figure()
        ax = plt.axes()

        min_val = min(self.x)
        max_val = max(self.x)
        test_range = np.arange(min_val - 1, max_val + 1, 0.1)
        num_test = len(test_range)
        #test_range.shape = (num_test, 1)

        preds = 
        preds = test_range @ self.opt_w
        ub = test_range @ self.opt_u
        lb = test_range @ self.opt_l
    
        ax.plot(self.x[:,0], self.predict, alpha=0.5, color='g', label = 'predict')
        ax.fill_between(self.x[:,0], self.lb, self.ub, facecolor='k', alpha=0.2)
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

