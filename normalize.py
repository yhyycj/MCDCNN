 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:34:09 2017

@author: yahuishi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def std_scale(dfs, tarCols):
    '''
    dfs: list of input data frames, the first one is used to fit the transformer
    '''
    for colName in tarCols:
        for df in dfs:
            df[colName].interpolate(limit_direction = 'both', inplace = True)
    stdScaler = StandardScaler()
    stdScaler.fit(dfs[0][tarCols])
    for i in range(len(dfs)):
        dfs[i][tarCols] = stdScaler.transform(dfs[i][tarCols])
    return dfs

def main():
    # load data
    outpath_train = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_train_noSeg.csv'
    outpath_valid = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_valid_noSeg.csv'
    outpath_test_pos = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_test_pos_noSeg.csv'
    outpath_test_neg = '../data/csv/for_DAD/noSeg/Fail_in_test/CF6_test_neg_noSeg.csv'
    
    df_train = pd.read_csv(outpath_train)
    df_valid = pd.read_csv(outpath_valid)
    df_test_pos = pd.read_csv(outpath_test_pos)
    df_test_neg = pd.read_csv(outpath_test_neg)
    
    # standardization
    tarCols = ['cr_degt', 'cr_gpcn25', 'cr_gwfm', 'cr_zpcn12', 'cr_zpcn25', 'cr_zps3', 'cr_zt49', 'cr_zvb1f', 'cr_zvb2r', 'cr_zwf36', 'tk_egthdm', 'tk_zpcn12', 'tk_zt49', 'tk_zvb1f', 'tk_zvb2r', 'tk_zwf36']
    df_train_std, df_valid_std, df_test_pos_std, df_test_neg_std = std_scale([df_train, df_valid, df_test_pos, df_test_neg], tarCols)
    
    # write to files
    for i in range(4):
        outfp = [outpath_train, outpath_valid, outpath_test_pos, outpath_test_neg][i]
        data = [df_train_std, df_valid_std, df_test_pos_std, df_test_neg_std][i]
        outfp = outfp.replace('.csv', '_intp_std.csv')
        data.loc[:,tarCols + ["Tag", "Offset", "gnt_key"]].to_csv(outfp, header = True, index = False)
    
if __name__ == '__main__':
    main()