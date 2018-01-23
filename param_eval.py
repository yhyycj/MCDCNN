#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace a parameter with 'Normal' data, and measure the drop in probability compared to the original prediction to evaluate the importancy of a parameter.


Created on Mon Dec 18 09:42:35 2017

@author: yahuishi
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from data_prepare import load_data_Xy, batch_generator
import matplotlib.pyplot as plt


def replace_sig(X1, X2, tar_index):
    '''
    X1, X2: nested list of parameters
    tar_index: list of target index to be replaced
    
    return X1' with the target sub-list replaced by data from X2
    '''
    # print("X1 length {}; X2 lenght {}".format(len(X1[0]), len(X2[0])))
    if not isinstance(tar_index, list):
        tar_index = [tar_index]
    X3 = []
    for i in range(len(X1)):
        if i in tar_index:
            if len(X2[i]) >= len(X1[i]):
                remain_len = len(X2[i]) - len(X1[i]) + 1
                index_start = np.random.randint(0, remain_len)
                X3.append(X2[i][index_start:index_start+len(X1[0])])   # cut a pice from head of X2 to replace X1
            else:
                keep_len = len(X1[i]) - len(X2[i])
                new_seq = X1[i][:keep_len] + X2[i]
                X3.append(new_seq)  # use X2 to replace the tail of X1
        else:
            X3.append(X1[i])
    return X3

def get_pred(data_X, checkpoint_file):
    tf.reset_default_graph()
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        prediction = graph.get_operation_by_name("Model/prediction").outputs[0]
        X = graph.get_tensor_by_name("X:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        rlts = []
        for x_batch_list in batch_generator(data_X, batch_len = 32, max_num = None):
            x_batch = np.swapaxes(np.array(x_batch_list).reshape([1, 16, 32]), 1, 2)
            feed_dict = {
                X: x_batch,
                keep_prob: 1.0
            }
            pred = sess.run(prediction, feed_dict)[0]
            rlts.append(pred[1])
        return rlts
    
def param_eval(abRlts, method = 'binary', abs_diff_threshold = 0.5):
    '''
    calculate parameter importancy as the percentage of points that
    abRlts: DataFrame, first column is the label ('origin' or parameter name that has been replaced)
    method: {'binary', 'numeric'}
    '''
    evalRlts = {}
    for i_row in range(abRlts.shape[0]):
        param = abRlts.iloc[i_row, 0]
        if param == 'origin':   # prediction on original test sample
            raw_preds = abRlts.iloc[i_row, 1:]
            if method == 'binary':
                raw_preds_binary = raw_preds >= 0.5
        else:   # prediction on test sample with one parameter replaced by 'normal' training sample
            preds = abRlts.iloc[i_row, 1:]
            if method == 'binary':
                preds_binary = preds >= 0.5
                diff_binary = preds_binary ^ raw_preds_binary
            elif method == 'numeric':
                abs_diff = abs(preds - raw_preds)
                diff_binary = abs_diff >= abs_diff_threshold    # count 1 when prediction differs larger than threshold after replacing the current parameter
            diff_ratio = diff_binary.sum() * 1.0 / diff_binary.shape[0]
            evalRlts[param] = diff_ratio
    return evalRlts

if __name__ == '__main__':
    # load data
    data_train = '../data/csv/for_MCDCNN/CF6_train.csv'
    data_valid = '../data/csv/for_MCDCNN/CF6_valid.csv'
    data_test = '../data/csv/for_MCDCNN/CF6_test.csv'
    
    ESN_train, X_train, y_train = load_data_Xy(data_train)  # Normal 7, Fail 3
    ESN_valid, X_valid, y_valid = load_data_Xy(data_valid)  # Normal 2, Fail 1
    ESN_test, X_test, y_test = load_data_Xy(data_test)  # Normal 3, Fail 2
    
    all_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'cr_zpcn12', 'cr_zpcn25', 'cr_zps3', 'cr_zt49', 'cr_zvb1f', 'cr_zvb2r', 'cr_zwf36', 'tk_egthdm', 'tk_zpcn12', 'tk_zt49', 'tk_zvb1f', 'tk_zvb2r', 'tk_zwf36']

    test_esn = 706902
    data_X = X_test[ESN_test.index(test_esn)]
    data_y = y_test[ESN_test.index(test_esn)]
    if data_y == [0, 1]:
        data_tag = 'Fail'
    else:
        data_tag = 'Normal'

    normal_esn = 706903
    normal_X = X_train[ESN_train.index(normal_esn)]
    
    checkpoint_file = '/Users/yahuishi/Documents/20171123_CF6_Blade/main/checkpoints/ratio05_batch80/model-700'
    
    outpath = checkpoint_file.replace('checkpoints', 'preds') + '-ablate/' + '_'.join([str(test_esn), str(normal_esn)]) + '.csv'
    if not os.path.exists('/'.join(outpath.split('/')[:-1])):
        os.mkdir('/'.join(outpath.split('/')[:-1]))
    
    # write predictions to local file, for further evaluation
    with open(outpath, 'w') as outfile:
        preds = get_pred(data_X, checkpoint_file)    # get original preds
        outfile.write(','.join([str(test_esn), data_tag, 'origin'] + [str(r) for r in preds]) + '\n')
        for i in range(len(all_params)):    # replace a parameter in each iteration and track the predictions
            # ablate one signal
            ab_param = all_params[i]
            X_ablate = replace_sig(data_X, normal_X, i)
            # get prediction
            preds = get_pred(X_ablate, checkpoint_file)
            outfile.write(','.join([str(test_esn), data_tag, ab_param] + [str(r) for r in preds]) + '\n')
        cmd = ' '.join(['rscript plot_preds_ablate.R', outpath, outpath.replace('.csv', '.pdf')])
        print(cmd)
        os.system(cmd)
        
    # evaluate parameter importancy
    # outpath = '/Users/yahuishi/Documents/20171123_CF6_Blade/main/preds/ratio05_batch80/model-700-ablate/706983_706903.csv'
    abRlts = pd.read_csv(outpath, sep = ',', header = None)
    evalRlts_bin = param_eval(abRlts.iloc[:,2:], method = 'binary')
    evalRlts_num = param_eval(abRlts.iloc[:,2:], method = 'numeric')
    with open(outpath.replace('.csv', '_paraEval.txt'), 'w') as outfile:
        outfile.write('# ------ {esnFail}_{esnNormal} ------\n'.format(esnFail = test_esn, esnNormal = normal_esn))
        outfile.write('# Parameter:\tbinary_importance\tnumeric_importance\n')
        for k, v in evalRlts_bin.items():
            outfile.write('{}:\t{:.4f}\t{:.4f}\n'.format(k, v, evalRlts_num[k]))
