from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pylab as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
import sklearn.kernel_approximation
import scipy.optimize as opt
from quantregression import QuantReg
from qreg import QRegressor

class UCB(ABC):
    """Base class for UCB.
    
    """
    def __init__(self, env, x):
        """
        Arguments
        ---------------------------------
        env: 
            instance of DummyEnvironment, generate samples using x
        x
            list of all index of samples
        """ 
        self.env = env
        self.x = x

        self.bestaction = self.x[np.argmax(
                            self.env.sample_withoutnoise(self.x))]
        self.X = []
        self.T = []
        self.cumu_regret = 0
        self.regret_list = []

    @abstractmethod
    def argmax_ucb(self):
        """compute upper bound.
        """

    def regret(self, t):
        """Compute regret and cumulative regret   
        """
        if self.X[-1] != self.bestaction:
            print('In ', t, 'iteration, selected: ', self.X[-1], ' best: ', self.bestaction)
            self.cumu_regret += self.env.sample(self.bestaction) - self.T[-1]
        self.regret_list.append(self.cumu_regret)

    def init_reward(self):
        for i in self.x:
            self.X.append(i)
            self.T.append(self.env.sample(i))

    @abstractmethod
    def learn(self, epoch):
        """sample index with maximum upper bound and 
           update gaussian process parameters
        """

    def sample(self, idx):
        """sample idx according to the gound truth
        """
        self.X.append(idx)
        self.T.append(self.env.sample(idx))

    @abstractmethod
    def plot(self):
        """Show cumulative regret vs. iteration
        """

class GPUCB(UCB):
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

    def __init__(self, env, x, alpha = 1., beta=1.):
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

        super().__init__(env, x)

        self.alpha = alpha
        self.beta = beta
        self.mu = np.zeros_like(x)
        self.sigma = 0.5 * np.ones_like(x)
        # self.gp = GaussianProcessRegressor(alpha = self.alpha)
        self.gp = GaussianProcessRegressor(kernel=DotProduct())

    def argmax_ucb(self):
        """compute upper bound.
        """
        return self.X[np.argmax(self.mu + self.sigma * self.beta)][0]

    def learn(self, epoch):
        """sample index with maximum upper bound and update gaussian process parameters
        """
        if len(self.X) > 0:   
            self.beta = 2.0 * np.log(len(self.X) * (epoch + 1.0))/20
        
        self.gp.fit(self.X, self.T)
        self.mu, self.sigma = self.gp.predict(self.x.reshape((self.x.shape[0],1)), return_std=True)

        idx = self.argmax_ucb()
        self.sample(idx)
        self.regret(epoch)

    def init_reward(self):
        for i in self.x:
            self.X.append([i])
            self.T.append(self.env.sample(i))

    def sample(self, idx):
        """sample idx according to the gound truth
        """
        self.X.append([idx])
        self.T.append(self.env.sample(idx))

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
        init_len = len(self.x)
        ax.scatter(self.X[:init_len], self.T[:init_len], c='b', marker='o', alpha=1.0, label = 'init sample')
        ax.scatter(self.X[init_len:], self.T[init_len:], c='r', marker='o', alpha=1.0, label = 'selected sample')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('GPUCB')
        plt.show()

class QuantUCB_MUB(UCB):
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

    def __init__(self, env, x):

        super().__init__(env, x)

        self.alpha = 0.5
        
        self.predict = np.zeros_like(self.x)
        
        self.reg = QRegressor(C=1e2, probs=[0.1,0.5,0.9], gamma_out=1e-2, max_iter=1e4,
                            verbose=False, lag_tol=1e-3, active_set=True)
        
    def argmax_ucb(self, t, num_rounds):
        
        self.alpha = 0.5 + np.log(t + 1)/(num_rounds * 2)
        return self.X[np.argmax(self.quantile)]

    def learn(self, t, num_rounds):
    
        #print(self.X)
        #print(self.T)
        #self.QuantReg.fit(self.X, self.T, self.uq, self.lq)
        self.reg.fit(self.X, self.T, [self.alpha])
        self.quantile = self.reg.predict(self.x)[0]
        #self.predict, self.ub, self.lb = self.QuantReg.predict(self.x)
        idx = self.argmax_ucb(t, num_rounds)
        self.sample(idx)
        self.regret(t)
   
    def plot(self):
        ax = plt.axes()

        min_val = min(self.x)
        max_val = max(self.x)
        test_range = np.arange(min_val - 1, max_val + 1, 0.1)
        #preds, ub, lb = self.QuantReg.predict(test_range)
        pred = self.reg.predict(test_range)

        ax.plot(test_range, pred[0], alpha=0.5, color='g', label = 'predict')
        #ax.fill_between(test_range, pred[0], pred[2], facecolor='k', alpha=0.2)
        init_len = len(self.x)
        ax.scatter(self.X[:init_len], self.T[:init_len], c='b', marker='o', alpha=1.0, label = 'init sample')
        ax.scatter(self.X[init_len:], self.T[init_len:], c='r', marker='o', alpha=1.0, label = 'selected sample')
        #plt.savefig('fig_%02d.png' % len(self.X))
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('QuantUCB')
        plt.show()
    
        
class DummyEnvironment(object):
    """Environment to generate samples with noise.
    """
    def rand_skew_norm(self, fAlpha, fLocation, fScale):
        sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

        afRN = np.random.randn(2)
        u0 = afRN[0]
        v = afRN[1]
        u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

        if u0 >= 0:
            return u1*fScale + fLocation 
        return (-u1)*fScale + fLocation 

    def sample(self, x):
        """Generating model is sin(x). 
        Noise model is normal distribution.
        """
        #noise = np.random.normal(0,0.1,1)
        noise = self.rand_skew_norm(0,0.1,4)
        return 10 * np.sin(x) + noise
        #return np.sin(x)
        
    def sample_withoutnoise(self, x):
        return 10 * np.sin(x)


#-------------------------------------------------------------------------------------------
# copy from UCB_discrete.py 
# base class can be found in that file

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

