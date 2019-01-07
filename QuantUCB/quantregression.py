"""Quantile Regression"""
import sklearn.kernel_approximation
import scipy.optimize as opt
import numpy as np

class QuantReg(object):
    """Quantile Regression
    """
    def __init__(self, D):

        self.D = D
        self.sampler = sklearn.kernel_approximation.RBFSampler(
                        n_components= self.D, gamma=0.1)

    def fit(self, x, t, uq,lq):
        x_ = np.asarray(x)
        x_ = np.stack([x_.ravel(), np.ones_like(x_.ravel())]).T
        x_rbf = self.sampler.fit_transform(x_)
        t_ = np.asarray(t)

        self.opt_w = opt.fmin_bfgs(self.cost, np.zeros((self.D,)), args=(x_rbf, t_, 0.5), disp= False)
        self.opt_u = opt.fmin_bfgs(self.cost, np.zeros((self.D,)), args=(x_rbf, t_, uq), disp = False)
        self.opt_l = opt.fmin_bfgs(self.cost, np.zeros((self.D,)), args=(x_rbf, t_, lq), disp = False)

    def cost(self, W, X, Y, q):
        predictions = X @ W
        return np.where(
            predictions > Y,
            (1 - q) * np.abs(Y - predictions),
            q * np.abs(Y - predictions)).sum()

    def predict(self, x):
        x_ = np.asarray(x)
        x_ = np.stack([x_.ravel(), np.ones_like(x_.ravel())]).T
        x_rbf = self.sampler.transform(x_)
        
        predict = x_rbf @ self.opt_w
        ub = x_rbf @ self.opt_u
        lb = x_rbf @ self.opt_l

        return predict, ub, lb

    