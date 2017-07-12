# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:58:19 2017

@author: Xiaoxi
"""

%reset

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats, linalg

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

import seaborn as sns; sns.set()


digits = load_digits()
X = digits.data
y = digits.target

plt.imshow(digits.images[0])

pca = PCA(2) # reduce from 64 to 2
Xproj = pca.fit_transform(X)

plt.scatter(Xproj[:,0], Xproj[:,1], c = y,
            edgecolor = "none", alpha =0.5,
            cmap = plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar            

'component ratio'
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_)) # to achieve 90% need 20 dims

'data compression'
fig, axes = plt.subplots(6,6, figsize = (8,8))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

for i,ax in enumerate(axes.flat):
    pca = PCA(i + 1).fit(X)
    im = pca.inverse_transform(pca.transform(X[9])) # 0
    
    ax.imshow(im.reshape(8,8),cmap = 'binary') # reshape ~ read to rows
    ax.text(0.95, 0.05, 'n = {0}'.format(i + 1), ha = 'right',
            transform = ax.transAxes, color = 'green')
            
    ax.set_xticks([])
    ax.set_yticks([])
        
