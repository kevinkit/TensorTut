# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:13:01 2017

@author: khoefle
"""


##Imports the Tensorflow library
import tensorflow as tf

#Define the constant, with a value, datatype, and a representive name
nc1 = tf.constant(10,tf.float32,name='const1')
nc2 = tf.constant(11,tf.float32,name='const2')

#Describe the addition
na = tf.add(nc1,nc2)


with tf.Session() as session:
    print(session.run(na))