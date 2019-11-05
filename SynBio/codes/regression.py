import numpy as np
from collections import defaultdict, OrderedDict
import operator
from codes.environment import Rewards_env
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class Regression():
    def __init__(self, model, data, embedding_method = 'onehot'):
        self.model = model
        self.my_env = Rewards_env(data, embedding_method)
        self.X = self.my_env.embedded
        self.Y = data[:, 1]

    def split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size = 0.2, random_state = 42)

    def train(self):
        self.split_data()
        self.model.fit(self.X_train, self.Y_train)
        self.train_predict = self.model.predict(self.X_train)
        self.test_predict = self.model.predict(self.X_test)

    def evaluate(self, print_flag = True, plot_flag = True):
        train_score = self.model.score(self.X_train, self.Y_train)
        test_score = self.model.score(self.X_test, self.Y_test)

        if print_flag:
            print('Model: ', str(self.model))
            print('Train score: ', train_score)
            print('Test score: ', test_score)
        if plot_flag:
            self.plot()

    def plot(self):        
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
    

        

   

    