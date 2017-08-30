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

from tensorflow.contrib.learn.python import SKCompat
from tqdm import tqdm
FILE = 'prices.xls'

#Read data
def readData(filename='prices.xls'):
    b = xlrd.open_workbook(filename,encoding_override="utf-8")
    sheet = b.sheet_by_index(0)
    x1 = np.asarray([sheet.cell(i,1).value for i in range(1,sheet.nrows)])
    x2 = np.asarray([sheet.cell(i,3).value for i in range(1,sheet.nrows)])
    y = np.asarray([sheet.cell(i,2).value for i in range(1,sheet.nrows)])
    
    return x1,x2,y




#Read the data
x1_d,x2_d, y_d=readData()
#Define the model - since there should be a linear dependency, one layer
#should be enough
with tf.name_scope('mySecondModel'):
    #Weights, initialzed with 0
    W1 = tf.Variable([30000.0],name='Weights1')
    W2 = tf.Variable([1.0],name='Weights2')
    x1_t = tf.convert_to_tensor(x1_d,name='x1',dtype=tf.float32)
    x2_t = tf.convert_to_tensor(x2_d,name='x2',dtype=tf.float32)   
    y_t = tf.convert_to_tensor(y_d,name='y')
    #bias initialzed with 0
    b = tf.Variable([10.0],name='bias')
    
    mul1 = tf.multiply(W1,x1_t,name='FirstMul')
    mul2 = tf.multiply(W2,x2_t,name='SecondMul')
    addition = tf.add(mul1,mul2,name='add')
    y = tf.add(addition,b,name='biasadd')
    
    
    
    
    
    #y = W1*x1_d + W2*x2_d + b;
    
#Define the graph for training
with tf.name_scope('myFirstTraining'):
    #Calculate loss
    
    y_t = tf.convert_to_tensor(y_d,name='target')
    l = tf.reduce_mean(tf.square(tf.subtract(y,y_d)),name='loss')
    
    tf.summary.scalar('loss',l)
    
    
    #Optimizing Algorithm
    opt = tf.train.GradientDescentOptimizer(0.00001)
    t = opt.minimize(l)
    
#learn!
summary_of_ops = tf.summary.merge_all()
print(summary_of_ops)


with tf.Session() as session:
    wr = tf.summary.FileWriter('./housing',session.graph)
    init = tf.global_variables_initializer()
    session.run(init)
    
    #Train
    for i in tqdm(range(9999)):
        #print(i)
        summary,_ = session.run([summary_of_ops,t])
        wr.add_summary(summary,i)
        
    #Test
    cW1,cW2,cB,cl = session.run([W1,W2,b,l])
    #print("Weights: %s bias: %s loss: %s" % (cW1,cW2,cB,cl))
    print(cW1)
    print(cW2)
    print(cB)
    print(cl)
wr.close()
    
    
#    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features,model_dir='./linear_estimator_2')
#    estimator.fit(input_fn=input_fn_train)
#    
#    estimator.evaluate(input_fn=input_fn_train)
#    
#    estimator.predict(x={'x1': 150, 'x2': 1})

    #estimator = tf.contrib.learn.LinearRegressor(feature_columns=features,model_dir='./linear_estimator_2')
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