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


#____________________________________________
# step4: load image data from MNIST
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
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)


# The final preprocessing step for the input data is to convert our data type 
# to float32 and normalize our data values to the range [0, 1].
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


#__________________________________________________
# step6: preprocess class labels for Keras

# And there's the problem. The y_train and y_test data are not split into 10 
# distinct class labels, but rather are represented as a single array with the 
# class values.

## Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test, 10)


#_______________________________________________________
# Step-7: define model architecture
# declare sequential model
model = Sequential()

# declare input layer	
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
# The input shape parameter should be the shape of 1 sample. In this case, it's 
# the same (28, 28, 1) that corresponds to  the (width, height, depth) 
# of each digit image.

# 32,3,3 They correspond to the number of convolution filters to use, 
# the number of rows in each convolution kernel, and the number of columns 
# in each convolution kernel, respectively.

model.output.shape

#Next, we can simply add more layers to our model like we're building legos:
model.add(Convolution2D(32,3,3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# The Dropout is a method for regularizing the model to prevent overfitting

# MaxPooling2D is a way to reduce the number of parameters in our model by 
# sliding a 2x2 pooling filter across the previous layer and taking the max
#  of the 4 values in the 2x2 filter

# So far, for model parameters, we've added two Convolution layers. 
# To complete our model architecture, let's add a fully connected layer 
# and then the output layer:
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

# For Dense layers, the first parameter is the output size of the layer. 
# Keras automatically handles the connections between layers.

# Note that the final layer has an output size of 10, corresponding to 
# the 10 classes of digits.

# Also note that the weights from the Convolution layers must be flattened 
# (made 1-dimensional) before passing them to the fully connected Dense layer.

#Now all we need to do is define the loss function and the optimizer, 
# and then we'll be ready to train it.

# __________________________________________________________
# step 8: Compile model
# When we compile the model, we declare the loss function and the 
# optimizer (SGD, Adam, etc.). 
# ref: https://keras.io/optimizers/
model.compile(loss = 'categorical_crossentropy', # 'mean_squared_error'
              optimizer = 'adam', # sgd
              metrics = ['accuracy'])
              

#____________________________________________________________
# step9: fit model on training data
model.fit(X_train, Y_train,
          batch_size = 32, nb_epoch=10, verbose = 1)              