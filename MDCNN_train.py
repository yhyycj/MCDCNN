#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:10:08 2017

@author: yahuishi
"""

import os
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from MCDCNN import MDCDNN
from data_prepare import load_data_Xy, build_batches

# Parameters
learning_rate = 0.001
training_epoch = 20
shuffle_data_each_epoch = True
#batch_size = 100
display_step = 10
evaluate_every = 10
checkpoint_every = 10
num_checkpoints = 50
data_format = "NHWC"
out_dir = '.'
# Network Parameters
X_shape = [None, 32, 16] # "NHWC" format:  [batch, in_width, in_channels]
conv_input_shape_1 = [None, 32, 1]
class_num = 2 # Normal and Failed

# Start training
with tf.Graph().as_default():
    # tf Graph input
    X = tf.placeholder(tf.float32, X_shape, name = "X")
    y = tf.placeholder(tf.float32, [None, class_num], name = "y")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    # Set parameters
    with tf.name_scope('params'):
        W = {
            # conv1d_1: 1*5*8
            'wc1': tf.Variable(tf.random_normal([5, 1, 8]), name = 'wc1'),    #[filter_width, in_channels, out_channels]
            # conv1d_2: 1*3*4
            'wc2': tf.Variable(tf.random_normal([3, 8, 4]), name = 'wc2'),
            # fully connected: 4*16 input, 8 final_feature_dim
            'wd1': tf.Variable(tf.random_normal([6*4*16, 16]), name = 'wd1'),
            # output: 2 classes
            'wout': tf.Variable(tf.random_normal([16, class_num]), name = 'wout')
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([8]), name = 'bc1'),
            'bc2': tf.Variable(tf.random_normal([4]), name = 'bc2'),
            'bd1': tf.Variable(tf.random_normal([16]), name = 'bd1'),
            'bout': tf.Variable(tf.random_normal([class_num]), name = 'bout')
        }

    # Construct model
    with tf.name_scope('Model'):
        scores = MDCDNN(X, W, biases, keep_prob)
        pred = tf.nn.softmax(scores, name = "prediction")
    with tf.name_scope('Loss'):
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = y))
        # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))   # log generates 'inf' when predict is close to 0
    with tf.name_scope('Optimize'):
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

        # keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

    with tf.name_scope('Accuracy'):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
    
    with tf.Session() as sess:
        # Create summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cost)
        acc_summary = tf.summary.scalar("accuracy", acc)
    
        # Initialize variables
        init = tf.global_variables_initializer()
    
        # load data
        data_train = '../data/csv/for_MCDCNN/CF6_train.csv'
        data_valid = '../data/csv/for_MCDCNN/CF6_valid.csv'
        data_test = '../data/csv/for_MCDCNN/CF6_test.csv'
    
        ESN_train, X_train, y_train = load_data_Xy(data_train)  # Normal 7, Fail 3
        ESN_valid, X_valid, y_valid = load_data_Xy(data_valid)  # Normal 2, Fail 1
        ESN_test, X_test, y_test = load_data_Xy(data_test)  # Normal 3, Fail 2
    
        all_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'cr_zpcn12', 'cr_zpcn25', 'cr_zps3', 'cr_zt49', 'cr_zvb1f', 'cr_zvb2r', 'cr_zwf36', 'tk_egthdm', 'tk_zpcn12', 'tk_zt49', 'tk_zvb1f', 'tk_zvb2r', 'tk_zwf36']
        # tar_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'tk_egthdm']
        tar_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'tk_egthdm', 'cr_zvb1f', 'tk_zt49']
        
    
        # train summaries
        # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train", "ratio06_batch80_sig6Parm")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
        # dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev", "ratio06_batch80_sig6Parm")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    
        # checkpoint directory. Tensorflow assumes this directory already exists so we need to create it.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints", "ratio06_batch80_sig6Parm"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = num_checkpoints)
        
        sess.run(init)
        
        # Training cycle
        for epoch in range(training_epoch):
            batches_train = build_batches(X_train, y_train, batch_size = 80, seq_length = 32, norm_fail_ratio = 0.6)
            for x_batch, y_batch in batches_train:
                feed_dict_train = {
                    X: x_batch, 
                    y: y_batch,
                    keep_prob: 0.75
                }
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cost, acc], feed_dict_train)
                current_step = tf.train.global_step(sess, global_step)
                if current_step < 10 or current_step % display_step == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("TRAIN {:d} positive samples".format(np.sum([1 for i in range(y_batch.shape[0]) if y_batch[i].tolist() == [0, 1]])))
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)
                
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step = current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                if current_step < 10 or current_step % evaluate_every == 0:
                    batches_test = build_batches(X_valid, y_valid, batch_size = 20, seq_length = 32, norm_fail_ratio = 0.5, max_batch_num = 3)
                    loss_mean = []
                    acc_mean = []
                    pos_y = 0
                    for x_test_batch, y_test_batch in batches_test:
                        feed_dict_eval = {
                            X: x_test_batch,
                            y: y_test_batch,
                            keep_prob: 1.0
                        }
                        pos_y += np.sum([1 for i in range(y_test_batch.shape[0]) if y_test_batch[i].tolist() == [0, 1]])
                        step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cost, acc], feed_dict_eval)
                        dev_summary_writer.add_summary(summaries, step)
                        loss_mean.append(loss)
                        acc_mean.append(accuracy)
                    time_str = datetime.datetime.now().isoformat()
                    print("TEST {:d} positive samples".format(int(pos_y)))
                    print("{}: step {}, loss: {:g}, acc {:g}".format(time_str, step, np.mean(loss_mean), np.mean(acc_mean)))
                    