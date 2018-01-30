# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:09:09 2018

@author: Xiaoxi
"""


import mnist_loader

data_folder = 'C:/Users/Xiaoxi/Dropbox/work/2017/Others/python/data/'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(data_folder)
training_data = list(training_data)