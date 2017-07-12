# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:17:41 2017

@author: Xiaoxi
"""

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1

import numpy as np
import keras


np.random.seed(123)


# Next, we'll import the Sequential model type from Keras. 
# This is simply a linear stack of neural network layers, 
# and it's perfect for the type of feed-forward CNN we're building
# in this tutorial.
from keras.models import Sequential

#Next, let's import the "core" layers from Keras. 
#These are the layers that are used in almost any neural network:
from keras.layers import Dense, Dropout, Activation, Flatten

#Then, we'll import the CNN layers from Keras. These are the convolutional layers that will 
# help us efficiently train on image data:
from keras.layers import Convolution2D, MaxPooling2D

# Finally, we'll import some utilities. This will help us transform our data later:
from keras.utils import np_utils


# load image data from MNIST
from keras.datasets import mnist

# load pre-shuffled mnist data into train and test
(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train.shape


# we have 60,000 samples in our training set, and the images
#  are 28 pixels x 28 pixels each
from matplotlib import pyplot as plt
plt.imshow(X_train[0])


# Preprocess data
#When using the Theano backend, you must explicitly declare a dimension 
#for the depth of the input image. For example, a full-color image with 
#all 3 RGB channels will have a depth of 3.

#Our MNIST images only have a depth of 1, but we must explicitly declare that.

#In other words, we want to transform our dataset from having 
# shape (n, width, height) to (n, depth, width, height).