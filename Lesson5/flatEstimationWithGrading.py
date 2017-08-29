# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:12:03 2017

@author: khoefle
"""

#Machine learning example
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd



FILE = 'prices.xls'

#Read data
def readData(filename='prices.xls'):
    b = xlrd.open_workbook(filename,encoding_override="utf-8")
    sheet = b.sheet_by_index(0)
    x1 = np.asarray([sheet.cell(i,1).value for i in range(1,sheet.nrows)])
    x2 = np.asarray([sheet.cell(i,3).value for i in range(1,sheet.nrows)])
    y = np.asarray([sheet.cell(i,2).value for i in range(1,sheet.nrows)])
    
    return x1,x2,y



with tf.device('/gpu:0'): #with tf.device('/cpu:0'):
    #Read the data
    x1_d,x2_d, y_d=readData()
    
    features = [tf.contrib.layers.real_valued_column("x1",dimension=1),\
                tf.contrib.layers.real_valued_column("x2",dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features,model_dir='./linear_estimator_2')
    input_fn_train = tf.contrib.learn.io.numpy_input_fn({"x1":x1_d,"x2":x2_d},y_d,num_epochs=1000)
    
    estimator.fit(input_fn=input_fn_train)
    
    estimator.evaluate(input_fn=input_fn_train)
    
    estimator.predict(x={'x1': 150, 'x2': 1})

#
##Define the model - since there should be a linear dependency, one layer
##should be enough
#with tf.name_scope('myFirstModel'):
#    #Weights, initialzed with 0
#    W = tf.Variable([0.0],name='Weights')
#    #bias initialzed with 0
#    b = tf.Variable([0.0],name='bias')
#    y = W*x_d + b;
#    
##Define the graph for training
#with tf.name_scope('myFirstTraining'):
#    #Calculate loss
#    l = tf.reduce_mean(tf.square(y - y_d),name='loss')
#    
#    tf.summary.scalar('loss',l)
#    
#    
#    #Optimizing Algorithm
#    opt = tf.train.GradientDescentOptimizer(0.001)
#    t = opt.minimize(l)
#    
##learn!
#summary_of_ops = tf.summary.merge_all()
#print(summary_of_ops)
#
#
#with tf.Session() as session:
#    wr = tf.summary.FileWriter('./housing',session.graph)
#    init = tf.global_variables_initializer()
#    session.run(init)
#    
#    #Train
#    for i in range(100000):
#        print(i)
#        summary,_ = session.run([summary_of_ops,t])
#        wr.add_summary(summary,i)
#        
#    #Test
#    cW,cB,cl = session.run([W,b,l])
#    print("Weights: %s bias: %s loss: %s" % (cW,cB,cl))
#    
#wr.close()
#        