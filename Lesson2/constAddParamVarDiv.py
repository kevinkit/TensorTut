# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:13:01 2017

@author: khoefle
"""


##Imports the Tensorflow library
import tensorflow as tf

with tf.device('/gpu:0'): #with tf.device('/cpu:0'):
    #Define the constant, with a value, datatype, and a representive name
    nc1 = tf.constant(10,tf.float32,name='const1')
    nc2 = tf.constant(11,tf.float32,name='const2')
    
    #Define the place holder x
    x = tf.placeholder(tf.float32,name='x')
    
    #Define the variable y
    y = tf.Variable(5.0,tf.float32,name='y')
    
    #Describe the subtraction
    ns = x - y;
    
    #Describe the addition
    na = tf.add(nc1,nc2)
    
    #Describe the divison
    nd = tf.divide(na,ns)
    
    #Create the Session object   
    with tf.Session() as session:
        #Initialize variables
        init = tf.global_variables_initializer()
        
        #parse to session
        session.run(init)
        print(session.run(nd,{x:7.0}))