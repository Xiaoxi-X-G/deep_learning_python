# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:49:58 2017

@author: Xiaoxi
"""

# ref: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/

import os 
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras
from keras.datasets import mnist

# stop potential reandomens
seed = 128
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('') # '..' means parent folder, '' current folder
data_dir = os.path.join(root_dir, 'data')
# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)

#
(X_train, y_train),(X_test, y_test) = mnist.load_data()