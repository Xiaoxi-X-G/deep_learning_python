# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:21:21 2017

@author: Xiaoxi
"""
%reset

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state




'create function to visu data'

colours = ['#4EACC5', '#FF9C34', '#4E9A06', '#F76A54']
palette = sns.color_palette('Set2', 10)

def plot_labels(X, labels):
    clusters = set(labels)
    if -1 in labels:
        clusters.remove(-1)
    print("There are {} clusters".format(len(clusters)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for cluster, colour in zip(clusters, colours):
        w = np.where(labels == cluster)[0]
        ax1.scatter(X[w,0], X[w,1], c = palette[cluster], marker = '.')
        
    if -1 in labels:
        print('There is noise in the dataset!')
        
        #plot with crosses
        w = np.where(labels == -1)[0]
        ax1.scatter(X[w,0],X[W,1], s = 400, c = 'r', marker = 'x')
        
    plt.show()    
       
       
### plot
random_state = check_random_state(14)     
X_orig, y_orig = make_blobs(n_samples = 1000,
                            cluster_std = 1, 
                            random_state = random_state)  
                            
plot_labels(X_orig, y_orig)                            