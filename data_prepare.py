#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:08:03 2017

@author: yahuishi
"""
import os
from itertools import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from normalize import std_scale

def read_raw_csv():
    os.chdir('/Users/yahuishi/Documents/20171123_CF6_Blade/main')
    df_records = pd.read_csv('../data/csv/CF6.csv')
    df_records['tk_flight_datetime'] = pd.to_datetime(df_records['tk_flight_datetime'])
#    df_records['tk_flight_datetime'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in df_records['tk_flight_datetime']]
    
    df_events = pd.read_csv('../data/csv/events.csv')
    df_events['On-wing'] = pd.to_datetime(df_events['On-wing'])
    df_events['Off-wing'] = pd.to_datetime(df_events['Off-wing'])
#    df_events['On-wing'] = [datetime.strptime(i, '%d/%m/%Y') for i in df_events['On-wing']]
#    df_events['Off-wing'] = [datetime.strptime(i, '%d/%m/%Y') for i in df_events['Off-wing']]
    
    df_params = pd.read_csv('../data/csv/params.csv')
    return df_records, df_events, df_params

def count_cycle_onwing2fail():
    df_events['CSO'] = pd.Series()
    for i_event in range(df_events.shape[0]):
        eng = df_events['ESN'][i_event]
        event_date = df_events['Off-wing'][i_event]
        onwing_date = df_events['On-wing'][i_event]
        if isinstance(onwing_date, pd.Timestamp):
            msk_eng = df_records['tk_esn'] == eng
            if isinstance(event_date, pd.Timestamp):
                msk_date = (df_records['tk_flight_datetime'] >= onwing_date) & (df_records['tk_flight_datetime']<= event_date)
            else:
                msk_date = df_records['tk_flight_datetime'] >= onwing_date
            msk = msk_eng & msk_date
            df_events.loc[df_events['ESN'] == eng, 'CSO'] = sum(msk)
            print("Enging {:g}: {:d}".format(eng, sum(msk)))
    return df_events

def generate_segs(df, record_length, key_pre = '', overlap = False):
    idx_start = 0
    df_segs = pd.DataFrame(columns = df.columns.tolist() + ['gnt_key'])
    while True:
        cut_len = int(np.random.normal(record_length, record_length * 0.1, 1))
        idx_end = min(idx_start + cut_len, df.shape[0])
        
        if idx_end - idx_start + 1 < 50:
            break
        
        seg = df.iloc[idx_start:idx_end, ]
        seg['gnt_key'] = '_'.join([key_pre, str(idx_start), str(cut_len)])
        seg['Offset'] = range(seg.shape[0])
        df_segs = pd.concat([df_segs, seg])
        if overlap == False:
            idx_start = idx_end
        else:
            pass    # TBD

    return df_segs

def write_to_file(data, filepath):
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f_new:
            data.to_csv(f_new, header = True, index = False)
    else:
        with open(filepath, 'a') as f_add:
            data.to_csv(f_add, header = False, index = False)

def build_input_RNN_noSeg(fail_in_train = False):
    df_records, df_events, df_params = read_raw_csv()
    
    outpath_train = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_train_noSeg.csv'
    outpath_valid = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_valid_noSeg.csv'
    outpath_test_pos = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_test_pos_noSeg.csv'
    outpath_test_neg = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_test_neg_noSeg.csv'

    tar_params = df_params.loc[df_params['prio'] == 1, 'parameter']
    if not fail_in_train:
        norm_ESNs = df_events.loc[df_events['Tag'] == 'Non-fail', 'ESN'].tolist()
        np.random.shuffle(norm_ESNs)
        ESN_train = norm_ESNs[:8]
        ESN_valid = norm_ESNs[8:10]
    else:
        print("TBC")
        pass
    for i_row in range(df_events.shape[0]):
        eng = df_events.loc[i_row, 'ESN']
        event_date = df_events.loc[i_row, 'Off-wing']
        onWing_date = df_events.loc[i_row, 'On-wing']
        flag_fail = df_events.loc[i_row, 'Tag']
        
        if not isinstance(onWing_date, pd.Timestamp):
            continue
        
        if isinstance(event_date, pd.Timestamp):
            msk_tw = (df_records['tk_flight_datetime'] >= onWing_date) & (df_records['tk_flight_datetime']<= event_date)
        else:
            msk_tw = df_records['tk_flight_datetime'] >= onWing_date
       
        msk_esn = df_records['tk_esn'] == eng
        msk = msk_esn & msk_tw
        df_records_eng_tw = df_records.loc[msk, tar_params.tolist() + ['Tag']]
        print(df_records_eng_tw.shape)
        
        if flag_fail == 'Non-fail':
            df_records_eng_tw['Tag'] = 'Non-fail'
            df_records_eng_tw['Offset'] = range(df_records_eng_tw.shape[0])
            df_records_eng_tw['gnt_key'] = eng
            if eng in ESN_train:
                write_to_file(df_records_eng_tw, outpath_train)
            elif eng in ESN_valid:
                write_to_file(df_records_eng_tw, outpath_valid)
            else:
                write_to_file(df_records_eng_tw, outpath_test_neg)
        elif flag_fail == 'Removed':
            if fail_in_train == True:
                df_records_eng_tw_normal = df_records_eng_tw.iloc[:-50,:]
                df_records_eng_tw_normal['Tag'] = 'Removed'
                df_records_eng_tw_normal['Offset'] = range(df_records_eng_tw_normal.shape[0])
                df_records_eng_tw_normal['gnt_key'] = eng
                if eng in ESN_train:
                    write_to_file(df_records_eng_tw_normal, outpath_train)
                elif eng in ESN_valid:
                    write_to_file(df_records_eng_tw_normal, outpath_valid)
                
                df_records_eng_tw_fail = df_records_eng_tw.iloc[-50:,:]            
                df_records_eng_tw_fail['Tag'] = 'Removed'
                df_records_eng_tw_fail['Offset'] = range(df_records_eng_tw_fail.shape[0])
                df_records_eng_tw_fail['gnt_key'] = eng
                write_to_file(df_records_eng_tw_fail, outpath_test_pos)                
            else:
                df_records_eng_tw['Tag'] = 'Removed'
                df_records_eng_tw['Offset'] = range(df_records_eng_tw.shape[0])
                df_records_eng_tw['gnt_key'] = eng
                write_to_file(df_records_eng_tw, outpath_test_pos)
                
def build_input_RNN():
    df_records, df_events, df_params = read_raw_csv()
    
    outpath_train = '../data/csv/for_DAD/CF6_train.csv'
    outpath_valid = '../data/csv/for_DAD/CF6_valid.csv'
    outpath_test_pos = '../data/csv/for_DAD/CF6_test_pos.csv'
    outpath_test_neg = '../data/csv/for_DAD/CF6_test_neg.csv'
    
    # average lenght of segments
    record_length = 100
    
    tar_params = df_params.loc[df_params['prio'] == 1, 'parameter']
    
    for i_row in range(df_events.shape[0]):
        eng = df_events.loc[i_row, 'ESN']
        event_date = df_events.loc[i_row, 'Off-wing']
        onWing_date = df_events.loc[i_row, 'On-wing']
        flag_fail = df_events.loc[i_row, 'Tag']
        
        if not isinstance(onWing_date, pd.Timestamp):
            continue
        
        if isinstance(event_date, pd.Timestamp):
            msk_tw = (df_records['tk_flight_datetime'] >= onWing_date) & (df_records['tk_flight_datetime']<= event_date)
        else:
            msk_tw = df_records['tk_flight_datetime'] >= onWing_date
        
        msk_esn = df_records['tk_esn'] == eng
        msk = msk_esn & msk_tw
        df_records_eng_tw = df_records.loc[msk, tar_params.tolist() + ['Tag']]
        print(df_records_eng_tw.shape)
            
        if flag_fail == 'Non-fail':
            df_records_eng_tw['Tag'] = 'Non-fail'
            
            idx_start = int(np.random.normal(5, 1, 1))
            df_segs = generate_segs(df_records_eng_tw.iloc[idx_start:], record_length, key_pre = str(eng), overlap = False)
            seg_keys = df_segs['gnt_key'].unique()
            seg_count = len(seg_keys)
            
            # train-valid-test split
            np.random.shuffle(seg_keys)
            keys_split_train = seg_keys[:int(seg_count*0.6)]
            keys_split_valid = seg_keys[int(seg_count*0.6):int(seg_count*0.8)]
            keys_split_test_neg = seg_keys[int(seg_count*0.8):]
            
            # write to file
            data_train = df_segs.loc[df_segs['gnt_key'].isin(keys_split_train),:]
            data_valid = df_segs.loc[df_segs['gnt_key'].isin(keys_split_valid),:]
            data_test_neg = df_segs.loc[df_segs['gnt_key'].isin(keys_split_test_neg),:]
            
            for i in range(3):
                data = [data_train, data_valid, data_test_neg][i]
                outpath = [outpath_train, outpath_valid, outpath_test_neg][i]
                write_to_file(data, outpath)
        
        elif flag_fail == 'Removed':
            df_records_eng_tw['Tag'] = 'Removed'
            idx_normal_start = int(np.random.normal(5, 1, 1))
            idx_fail_start = df_records_eng_tw.shape[0] - 50
            
            df_normal_segs = generate_segs(df_records_eng_tw.iloc[idx_normal_start:idx_fail_start,:], record_length, key_pre = str(eng), overlap = False)
            
            # train-valid-test split annd write to file
            seg_keys = df_normal_segs['gnt_key'].unique()
            seg_count = len(seg_keys)
            np.random.shuffle(seg_keys)
            keys_split_train = seg_keys[:int(seg_count*0.6)]
            keys_split_valid = seg_keys[int(seg_count*0.6):int(seg_count*0.8)]
            keys_split_test_neg = seg_keys[int(seg_count*0.8):]
            
            data_train = df_normal_segs.loc[df_normal_segs['gnt_key'].isin(keys_split_train),:]
            data_valid = df_normal_segs.loc[df_normal_segs['gnt_key'].isin(keys_split_valid),:]
            data_test_neg = df_normal_segs.loc[df_normal_segs['gnt_key'].isin(keys_split_test_neg),:]
            
            for i in range(3):
                data = [data_train, data_valid, data_test_neg][i]
                
                # add offset for ordering
#                data['Offset'] = range(data.shape[0])
                outpath = [outpath_train, outpath_valid, outpath_test_neg][i]
                write_to_file(data, outpath)
            
            # save 'failure' records to test
            df_fail = df_records_eng_tw.iloc[idx_fail_start:,:]
            df_fail['gnt_key'] = '_'.join([str(eng), str(idx_fail_start), '50'])
            df_fail['Offset'] = range(50)
            write_to_file(df_fail, outpath_test_pos)
            
        elif flag_fail == 'Suspect':
            pass     

def build_input_CNN(write2file = False):
    # load data
    df_records, df_events, df_params = read_raw_csv()
    df_events.dropna(subset = ['On-wing'], inplace = True)
    df_events.reset_index(drop = True, inplace = True)  # reset row number
    
    
    # output paths
    outpath_train = '../data/csv/for_MCDCNN/CF6_train.csv'
    outpath_valid = '../data/csv/for_MCDCNN/CF6_valid.csv'
    outpath_test = '../data/csv/for_MCDCNN/CF6_test.csv'
    
    tar_params = df_params.loc[df_params['prio'] == 1, 'parameter'].tolist()
    df_records = std_scale([df_records], tar_params)[0]
    
    ESN_normal = df_events.loc[df_events['Tag'] == 'Non-fail', 'ESN'].tolist()
    N_normal = len(ESN_normal)
    ESN_fail = df_events.loc[df_events['Tag'] == 'Removed', 'ESN'].tolist()
    N_fail = len(ESN_fail)
    
    # train/test split
    np.random.shuffle(ESN_normal)
    np.random.shuffle(ESN_fail)
    ESN_train = ESN_normal[:int(N_normal*0.6)] + ESN_fail[:int(N_fail*0.6)]
    ESN_valid = ESN_normal[int(N_normal*0.6):int(N_normal*0.8)] + ESN_fail[int(N_fail*0.6):int(N_fail*0.8)]
    ESN_test = ESN_normal[int(N_normal*0.8):] + ESN_fail[int(N_fail*0.8):]
    print("Normal: {:d} train/ {:d} valid/ {:d} test\nFail: {:d} train/ {:d} valid/ {:d} test".format(int(N_normal*0.6), int(N_normal*0.8)-int(N_normal*0.6), N_normal - int(N_normal*0.8), int(N_fail*0.6), int(N_fail*0.8)-int(N_fail*0.6), N_fail - int(N_fail*0.8)))
    
    for i_row in range(df_events.shape[0]):
        eng = df_events.loc[i_row, 'ESN']
        event_date = df_events.loc[i_row, 'Off-wing']
        onWing_date = df_events.loc[i_row, 'On-wing']
        flag_fail = df_events.loc[i_row, 'Tag']
        
        if isinstance(event_date, pd.Timestamp):
            msk_tw = (df_records['tk_flight_datetime'] >= onWing_date) & (df_records['tk_flight_datetime']<= event_date)
        else:
            msk_tw = df_records['tk_flight_datetime'] >= onWing_date
        
        msk_esn = df_records['tk_esn'] == eng
        msk = msk_esn & msk_tw
        df_records_eng_tw = df_records.loc[msk, ['tk_esn'] + tar_params]
        print(df_records_eng_tw.shape)
        
        df_records_eng_tw['Tag'] = flag_fail
        
        if write2file:
            if eng in ESN_train:
                outpath = outpath_train
            elif eng in ESN_valid:
                outpath = outpath_valid
            elif eng in ESN_test:
                outpath = outpath_test
            else:
                continue
            write_to_file(df_records_eng_tw, outpath)
            
def load_data_Xy(csvPath, tar_params = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'cr_zpcn12', 'cr_zpcn25', 'cr_zps3', 'cr_zt49', 'cr_zvb1f', 'cr_zvb2r', 'cr_zwf36', 'tk_egthdm', 'tk_zpcn12', 'tk_zt49', 'tk_zvb1f', 'tk_zvb2r', 'tk_zwf36']):
    df = pd.read_csv(csvPath)
    keys = df['tk_esn'].unique().tolist()
    y = df['Tag']
    vlist = []
    ylist = []
    for eng in keys:
        tag = df.loc[df['tk_esn'] == eng, 'Tag'].tolist()[0]
        vals = []
        if tag == 'Removed':
            ylist.append([0, 1])
        elif tag == 'Non-fail':
            ylist.append([1, 0])
        for fea in tar_params:
            vals.append(df.loc[df['tk_esn'] == eng, fea].tolist())
        vlist.append(vals)
    return keys, vlist, ylist

def build_batches(X_train, y_train, batch_size = 100, seq_length = 32, norm_fail_ratio = 0.5, max_batch_num = None):
    '''
    X_train, y_train: nested list of values, the first dimension is ESN
    '''
    normal_index = [i for i in range(len(y_train)) if y_train[i] == [1, 0]]
    fail_index = [i for i in range(len(y_train)) if y_train[i] == [0, 1]]
    np.random.shuffle(normal_index)
    
    
    X_batches = []
    y_batches = []
    batch_num = 0
    for i_sample in normal_index:
        for i_iter in range(int((len(X_train[i_sample][0]) - 500) - seq_length + 1)):
            # print("Normal sample({:d} index:{:d}-{:d})".format(i_sample, i_iter*seq_length, (i_iter+1)*seq_length))
            vals = []
            if i_iter*5 > len(X_train[i_sample][0]) or i_iter*5+seq_length > len(X_train[i_sample][0]):
                break   # end of the sequence
            for i_fea in range(16):
                vals.extend(X_train[i_sample][i_fea][i_iter*5:i_iter*5+seq_length])
            X_batches.extend(vals[:])
            y_batches.append(y_train[i_sample])
            if len(y_batches) == int(batch_size * norm_fail_ratio):
                np.random.shuffle(fail_index)
                for j_sample in cycle(fail_index):
                    sample_total_length = len(X_train[j_sample][0])
                    # print(j_sample, sample_total_length, 50 - seq_length + 1)
                    for j_iter in range(50 - seq_length + 1):
                        # print("Failed sample({:d} index:{:d}-{:d})".format(j_sample, sample_total_length - 50 + (j_iter*seq_length), sample_total_length - 50 + ((j_iter+1)*seq_length)))
                        vals = []
                        for j_fea in range(16):
                            vals.extend(X_train[j_sample][j_fea][sample_total_length - 50 + j_iter: sample_total_length - 50 + j_iter + seq_length])
                        X_batches.extend(vals[:])
                        y_batches.append(y_train[j_sample])
                        if len(y_batches) == batch_size:
                            shuffled_index = np.random.permutation(range(batch_size))
                            X_batches = np.swapaxes(np.array(X_batches).reshape([batch_size, 16, seq_length]), 1, 2)    # return shape: [batch_size, sequence_length, feature_num]
                            y_batches = np.array(y_batches).reshape([-1, 2])
                            yield X_batches[shuffled_index], y_batches[shuffled_index]
                            batch_num += 1
                            if max_batch_num is not None and batch_num == max_batch_num:
                                return
                            else:
                                X_batches = []
                                y_batches = []
                                break
                    if y_batches == []:
                        break
                            
def batch_generator(sequence, batch_len, max_num = None):
    batch_num = 0
    for i_index in range(len(sequence[0])-batch_len+1):
        vals = []
        for i_fea in range(len(sequence)):
            vals.append(sequence[i_fea][i_index:i_index+batch_len])
        yield vals
        batch_num += 1
        if max_num is not None and batch_num == max_num:
            return
        
if __name__ == '__main__':
#    df_records, df_events, df_params = read_raw_csv()
    
    # build input data
#    outpath_train = '../data/CF6_train.csv'
#    outpath_valid = '../data/CF6_valid.csv'
#    outpath_test_pos = '../data/CF6_test_pos.csv'
#    outpath_test_neg = '../data/CF6_test_neg.csv'
#    record_length = 1000
#    
#    tar_params = df_params.loc[df_params['prio'] == 1, 'parameter']
#    
#    df_events = count_cycle_onwing2fail()
#    df_events.to_csv('../data/csv/events.csv', index = False)
#    
#    build_input_RNN()
#    build_input_RNN_noSeg(fail_in_train = False)
#    
#    build_input_CNN(write2file = True)

    keys, X, y = load_data_Xy('../data/csv/for_MCDCNN/CF6_train.csv')
    # for x_batch, y_batch in build_batches(X, y, batch_size = 100, seq_length = 32, norm_fail_ratio = 0.5):
    #     x_batch = np.swapaxes(np.array(x_batch).reshape([100, 16, 32]), 1, 2)
    #     y_bath = np.array(y_batch).reshape([-1, 2])
    #     break

    print(len(X[0]))
    i = 0
    for x_batch in batch_generator(X[0], 32, None):
        i += 1
        print(i)
    
    
            
        