# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:37:29 2017

@author: khoefle
"""


from PIL import Image
import numpy as np

#Include the example
from tensorflow.examples.tutorials.mnist import input_data

#Download example and encode labels with ONE-HOT Coding
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


def debugData(part,idx=10):


    #Look at an example image
    x = part.images[idx]
    x = x.reshape([28,28])

    y = part.labels[idx]

    return x,y




#Get random example data
x_train,y_train = debugData(mnist.train,idx=np.random.randint(0,5000))
x_val,y_val = debugData(mnist.validation,idx=np.random.randint(0,5000))
x_test,y_test  = debugData(mnist.test,idx=np.random.randint(0,5000))



#Show the example data
Image.fromarray(np.asarray(x_train*255,dtype=np.uint8))
print(y_train)    

Image.fromarray(np.asarray(x_val*255,dtype=np.uint8))
print(y_val)    

Image.fromarray(np.asarray(x_test*255,dtype=np.uint8))
print(y_test)    