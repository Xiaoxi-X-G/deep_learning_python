# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:06:25 2017

@author: Xiaoxi

Ref: http://scikit-learn.org/stable/modules/sgd.html

Stochastic gradient descent (SGD) learning is an 
efficient approach to discriminative learning of linear classifiers under 
convex loss functions such as (linear) Support Vector Machines and Logistic 
Regression

"""

%reset

%matplotlib inline

import seaborn; seaborn.set() # set seaborn plot defaults
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier #stochastic gradient descent (SGD) learning
from sklearn.datasets.samples_generator import make_blobs

from sklearn.linear_model import LinearRegression



def plot_sgsd_separator():
    
    # return X ~ sample, Y~cluster index
    X,Y = make_blobs(n_samples = 50, # tot number of points equally divided among cluster
                     centers = 2, # number of centers to generate
                     random_state = 0,  #
                     cluster_std = 1) # std of cluster
                     
    #fit the model
    # SGD: similarly as using subgradent to find lagrangen multiplier, with each 
    # data set as a channel set (h1,h2)
    clf = SGDClassifier(loss = "hinge", #(soft-margin) linear Support Vector Machine
                        alpha = 0.01,  # step size
                        n_iter = 200, 
                        fit_intercept = True)    
                        
                        
    clf.fit(X,Y)

    #plot
    xx = np.linspace(-2,10,5)
    yy = np.linspace(-2,10,5)
    
    X1,X2 = np.meshgrid(xx,yy) # create grid according to inputs
    Z = np.empty(X1.shape)
    
    for (i, j), val in np.ndenumerate(X1): # multi-dim index interator, return index and value pair
        x1 = val
        x2 = X2[i, j]
        print(x1,x2)
        p = clf.decision_function(np.array([[x1, x2]])) # Distance of the samples X to the separating hyperplane
        Z[i, j] = p[0]
        
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors ='k'     
        
    ax = plt.axes()
    ax.contour(X1, X2, Z, levels, colors = colors, linestyles = linestyles)
    ax.scatter(X[:,0], X[:,1], c = Y, cmap = plt.cm.Paired)
     
if __name__ == '__main__'':
    plot_sgd_separator()
    plt.show()
    
    
                
                                 
                     