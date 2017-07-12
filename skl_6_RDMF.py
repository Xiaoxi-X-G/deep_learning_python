# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:32:27 2017

@author: Xiaoxi
"""

% reset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats

import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier


def plot_decision_regions(X, y, classifier, resolution = 0.02):
    
    #setup marker generateor and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[: len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x2_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
                   
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]))
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, alpha=0.4, cmap = cmap)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl, 0],
                    y = X[y==cl, 1],
                    alpha=0.8, c=cmap(idex),
                    marker = markers[idx], label = cl)
               
    


'start with decision tree'
X, y = make_blobs(n_samples = 300, centers = 4,
                  random_state = 0, cluster_std = 1.5)
 
plt.scatter(X[:,0], X[:,1], c = y, s= 50, cmap = "rainbow")       

clf = DecisionTreeClassifier(X,y) 

plot_decision_regions(X,y, classifier = clf)         