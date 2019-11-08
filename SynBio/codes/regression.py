import numpy as np
from collections import defaultdict, OrderedDict
import operator
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from codes.environment import Rewards_env
from codes.kernels import spectrum_kernel, mixed_spectrum_kernel, WD_kernel, WD_shift_kernel

KERNEL_TYPE = {'spectrum': spectrum_kernel,
               'mixed_spectrum': mixed_spectrum_kernel,
               'WD': WD_kernel,
               'WD_shift': WD_shift_kernel
            }

class Regression():
    """Regression.

    Attributes
    --------------------------------------------------------
    model: instance of regression class (from sklearn)
            attribute: kernel (if 'precomputed', use precomputed kernel matrix)
    X: array
        features array (num_samples, ) 
        first column of data, each element is a string
    Y: array
        labels array (num_samples, )
        second column of data, each element is a float/int

    TODO: add cross validation 
    """
    def __init__(self, model, data, embedding_method = None, 
                 precomputed_kernel = None):
        """
        Paramters:
        ------------------------------------------------------
        model: instance of regression class (from sklearn)
            attribute: kernel (if 'precomputed', use precomputed kernel matrix)
        data: ndarray 
            num_data * 2
            two columns: biology sequence; score (label)
        embedding_method: string, default is None
            if None, no embedding is performed and set X to the first column of data
        precomputed_kernel: string, default is None
            must be the key of KERNEL_TYPE dict
        """
        self.model = model
        
        if embedding_method is not None:
            self.my_env = Rewards_env(data, embedding_method)
            self.X = self.my_env.embedded
        else:
            self.X = data[:, 0]
        self.Y = data[:, 1]
        self.split_data()

        if self.model.kernel == 'precomputed':
            self.precompute_kernel(precomputed_kernel)

        if hasattr(self.model.kernel, 'metric'):
            if self.model.kernel.metric == 'precomputed':
                self.precompute_kernel(precomputed_kernel)
            

    def precompute_kernel(self, precomputed_kernel):
        # compute kernel matrix 
        # K_train n_train_samples * n_train_samples
        assert precomputed_kernel in KERNEL_TYPE
        kernel = KERNEL_TYPE[precomputed_kernel]
        K_train = kernel(self.X_train)
        # K_test n_test_samples * n_train_samples
        K_test = kernel(self.X_test, self.X_train)

        self.X_train = K_train
        self.X_test = K_test

    def split_data(self):
        """Split data into training and testing datasets. 
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size = 0.2, random_state = 42)

    def train(self):
        """Train.
        """
        self.model.fit(self.X_train, self.Y_train)
        self.train_predict = self.model.predict(self.X_train)
        self.test_predict = self.model.predict(self.X_test)

    def evaluate(self, print_flag = True, plot_flag = True):
        """Evaluate.
        Calculate R2 score for both training and testing datasets.
        """
        train_score = self.model.score(self.X_train, self.Y_train)
        test_score = self.model.score(self.X_test, self.Y_test)

        if print_flag:
            print('Model: ', str(self.model))
            print('Train score: ', train_score)
            print('Test score: ', test_score)
        if plot_flag:
            self.plot()

    def plot(self):    
        """Plot for predict vs. true label. 
        """    
        plt.plot(self.test_predict, self.Y_test, 'r.', label = 'test')
        plt.plot(self.train_predict, self.Y_train, 'b.', label = 'train')
        plt.plot([0,2], [0,2], '--')
        plt.plot([0,2], [1,1], 'k--')
        plt.plot([1,1], [0,2], 'k--')
        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.title(str(self.model))
        plt.xlim(0,2)
        plt.ylim(0,2)
        plt.legend()
    

        

   

    