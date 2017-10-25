# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:34:28 2017

@author: Xiaoxi
"""

# http://neuralnetworksanddeeplearning.com/chap1.html

import network
from keras.datasets import mnist

training_data_old, test_data_old = mnist.load_data()
training_data = list(zip(training_data_old[0], training_data_old[1]))
test_data = list(zip(test_data_old[0], test_data_old[1]))

net = network.Network([50,3,2])

net.SGD(training_data = training_data,
        epochs = 30, 
        mini_batch_size=10, 
        eta=3, 
        test_data = test_data)
         
        
                                
                                
                            
        