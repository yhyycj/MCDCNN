#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:31:00 2017

@author: yahuishi
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from data_prepare import load_data_Xy, batch_generator
import matplotlib.pyplot as plt
from itertools import *

all_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'cr_zpcn12', 'cr_zpcn25', 'cr_zps3', 'cr_zt49', 'cr_zvb1f', 'cr_zvb2r', 'cr_zwf36', 'tk_egthdm', 'tk_zpcn12', 'tk_zt49', 'tk_zvb1f', 'tk_zvb2r', 'tk_zwf36']
# tar_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'tk_egthdm'] # 4 significant parameters identified by plots
tar_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'tk_egthdm', 'cr_zvb1f', 'tk_zt49']  # 6 significant parameters identified by plots

tf.reset_default_graph()
checkpoint_file = '/Users/yahuishi/Documents/20171123_CF6_Blade/main/checkpoints/ratio05_batch80_sig6Parm/model-100'

# load data
data_train = '../data/csv/for_MCDCNN/CF6_train.csv'
data_valid = '../data/csv/for_MCDCNN/CF6_valid.csv'
data_test = '../data/csv/for_MCDCNN/CF6_test.csv'

ESN_train, X_train, y_train = load_data_Xy(data_train)  # Normal 7, Fail 3
ESN_valid, X_valid, y_valid = load_data_Xy(data_valid)  # Normal 2, Fail 1
ESN_test, X_test, y_test = load_data_Xy(data_test)  # Normal 3, Fail 2


with tf.Session() as sess:
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
  saver.restore(sess, checkpoint_file)
  # sess.run(tf.global_variables_initializer())
  prediction = graph.get_operation_by_name("Model/prediction").outputs[0]
  X = graph.get_tensor_by_name("X:0")
  keep_prob = graph.get_tensor_by_name("keep_prob:0")
  #X = tf.placeholder(tf.float32, [None, 32, 16])
  #keep_prob = tf.placeholder(tf.float32)
  outfile = open(checkpoint_file.replace('checkpoints', 'preds') + '_test.csv', 'w')
  colors = ['r', 'b', 'g', 'k', 'm']
  for i_eng in range(len(ESN_test)):
    plt.figure()
    tag = y_test[i_eng]
    rlts = [ESN_test[i_eng], 'Fail' if tag == [0,1] else 'Normal']
    for x_batch_list in batch_generator(X_test[i_eng], batch_len = 32, max_num = None):
        x_batch = np.swapaxes(np.array(x_batch_list).reshape([1, 16, 32]), 1, 2)
        feed_dict = {
            X: x_batch,
            keep_prob: 1.0
        }
        pred = sess.run(prediction, feed_dict)[0]
        rlts.append(pred[1])
    plt.scatter(x = range(len(rlts)-2), y = rlts[2:], color = 'r' if tag == [0, 1] else 'b', marker = 'o' if tag == [1, 0] else 'x', label = rlts[0])
    plt.legend()
    plt.title('Predictions on Test Set')
    plt.savefig(checkpoint_file.replace('checkpoints', 'preds') + '_' + str(ESN_test[i_eng]) + '.png', dpi = 300)
    outfile.write(','.join([str(r) for r in rlts]) + '\n')
  outfile.close()
#  plt.legend()
#  plt.savefig(checkpoint_file.replace('checkpoints', 'preds') + '.png', dpi = 300)
#  plt.show()  