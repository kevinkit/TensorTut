"""
This script will do a simple matrix multiplication

It will be executed on the CPU

"""
import tensorflow as tf with tf.device('/cpu:0'): 
   a = tf.constant([4.0, 4.0, 5.0, 2.0, 3.0, 5.0], shape=[2, 3], name='a') 
   b = tf.constant([5.0, 6.0, 6.0, 1.0, 3.0, 4.0], shape=[3, 2], name='b') 
   c = tf.matmul(a, b) 
   with tf.Session() as sess: print (sess.run(c)) 
