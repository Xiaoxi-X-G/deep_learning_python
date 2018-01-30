# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:34:28 2017

@author: Xiaoxi
"""

# http://neuralnetworksanddeeplearning.com/chap1.html
# https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/mnist_loader.py
import network
from keras.datasets import mnist
import mnist_loader

import pickle
import gzip

#tr_d,  te_d = mnist.load_data()
#training_data, test_data = network.load_data_wrapper(tr_d, te_d)
#
#training_data = list(training_data)



data_folder = 'C:/Users/Xiaoxi/Dropbox/work/2017/Others/python/data/'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(data_folder)
training_data = list(training_data)

net = network.Network([784,30,10])

net.SGD(training_data = training_data,
        epochs = 30, 
        mini_batch_size=10, 
        eta=3,
        test_data = test_data)
         
        
                                
                                
                            
        