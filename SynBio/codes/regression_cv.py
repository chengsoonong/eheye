import numpy as np
from collections import defaultdict, OrderedDict
import operator
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

from codes.environment import Rewards_env
from codes.kernels import spectrum_kernel, mixed_spectrum_kernel, WD_kernel, WD_shift_kernel


class Regression_cv():
    """Regression with cross validation (KFold).
    Also without precomputed kernel (regression.py support that). 

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

    def run_model(self, k =10):
        kf = KFold(n_splits= k, shuffle=True, random_state=42)
        train_scores = []
        test_scores = []
        init_model = self.model
        
        for (train_idx, test_idx) in kf.split(self.X):
            #print(train_idx)
            #print(test_idx)
            my_model = init_model
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]
            my_model.fit(X_train, Y_train)

            train_predict = my_model.predict(X_train)
            test_predict = my_model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
            test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))
            train_scores.append(train_rmse)
            test_scores.append(test_rmse)

            # train_score = my_model.score(X_train, Y_train)
            # test_score = my_model.score(X_test, Y_test)
            # train_scores.append(train_score)
            # test_scores.append(test_score)

            # if test_score < 0.4:
                
            plt.figure() 
            plt.scatter(np.abs(train_predict - Y_train), Y_train, color = 'r', label = 'train')
            plt.scatter(np.abs(test_predict - Y_test), Y_test, color = 'b', label = 'test')
            
            plt.title('Test RMSE: ' + str(test_rmse))
            plt.xlabel('|Predicted FC - FC|')
            plt.ylabel('FC')
            plt.legend()

        print('Model: ', str(self.model))
        train_scores = np.asarray(train_scores)
        test_scores = np.asarray(test_scores)
        print('train RMSE: ', train_scores)
        print('test RMSE: ', test_scores)

        print("Train RMSE: %0.2f (+/- %0.2f)" % (train_scores.mean(), train_scores.std() * 2))
        print("Test RMSE: %0.2f (+/- %0.2f)" % (test_scores.mean(), test_scores.std() * 2))
        

    

   
    

        

   

    