#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:22:09 2017

@author: yahuishi
"""

import tensorflow as tf
import pandas as pd
import numpy as np

def MDCDNN(X, W, biases, dropout):
    feature_num = 16
    
    pooled_outputs = []
    for i in range(feature_num): # feature_num
        with tf.name_scope("conv-maxpool-%s" % i):
            x = tf.reshape(X[:,:,i], [-1, 32, 1])
            conv1 = tf.nn.conv1d(x, W['wc1'], stride = 1, padding = 'VALID', name = "conv1d_1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']), name = "h1_relu")
            h1 = tf.reshape(h1, shape = [-1, 1, 28, 8])
            avgpool1 = tf.nn.avg_pool(h1, ksize = [1, 1, 2, 1], padding = 'VALID', strides = [1, 1, 2, 1], name = "avg_pool_1")
            avgpool1 = tf.reshape(avgpool1, shape = [-1, 14, 8])

            conv2 = tf.nn.conv1d(avgpool1, W['wc2'], stride = 1, padding = 'VALID', name = "conv1d_2")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
            h2 = tf.reshape(h2, shape = [-1, 1, 12, 4])
            avgpool2 = tf.nn.avg_pool(h2, ksize = [1, 1, 2, 1], strides = [1, 1, 2, 1], padding = "VALID", name = "avg_pool_2")
            avgpool2 = tf.reshape(avgpool2, shape = [-1, 1, 6, 4])
            pooled_outputs.append(avgpool2)
    # Combine all the pooled features
    num_filters_total = feature_num * 4
    h_pool = tf.concat(pooled_outputs, axis = 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total*6])
    
    # Fully connected layer
    fc1 = tf.nn.xw_plus_b(h_pool_flat, W['wd1'], biases['bd1'], name = "dense1")
    h_dense = tf.nn.relu(fc1)

    # Apply dropout
    h_drop = tf.nn.dropout(h_dense, dropout, name = "dropout")

    # Final score and prediction
    h_score = tf.nn.xw_plus_b(h_drop, W['wout'], biases['bout'], name = "scores")
    return(h_score)
