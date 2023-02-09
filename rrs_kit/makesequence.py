import os 

import pickle 
import numpy as np 
import pandas as pd 

from rrs_kit.DataClass import DataPath, VarSet 
from utils import (
    adjust_cbc,
    adjust_chem,
    abn_nl_concat, 
    make_2d_time,
    make_2d_data,
    make_sequence_data
)

dp = DataPath()
#train_whole, valid_whole = train_valid_split(sequence_data_train, 0.7)

def make_sequence_train():
    parse_dates = ['adjusted_time']

    trn_abn = pd.read_csv(os.path.join(dp.output_path, 'trn_abn_merged.csv'), parse_dates = parse_dates)
    trn_nl = pd.read_csv(os.path.join(dp.output_path, 'trn_nl_merged.csv'), parse_dates = parse_dates)

    trn_nl = trn_nl.rename({
        '성별': 'Gender', 
        '나이':'Age',
        '혈압_수축기': 'SBP', 
        '체온':'BT',
        '맥박':'HR', 
        '호흡':'RR', 
        '혈액검사시점' : 'blood_time',
        'WBC count' : 'WBC Count'}, axis=1)
    trn_nl = trn_nl.drop(['생년월일', 'datetime'], axis = 1)

    trn_abn = trn_abn.rename({
        '성별': 'Gender', 
        '나이':'Age',
        '혈압_수축기': 'SBP', 
        '체온':'BT',
        '맥박':'HR', 
        '호흡':'RR', 
        '혈액검사시점' : 'blood_time',
        'WBC count' : 'WBC Count'}, 
        axis=1)
    trn_abn = trn_abn.drop(['event','detection', 'datetime'], axis = 1)

    trn_abn['is_abn'] = 1
    trn_nl['is_abn'] = 0


    parse_dates = ['adjusted_time']
    tst_abn = pd.read_csv(os.path.join(dp.output_path, 'tst_abn_merged.csv'), parse_dates = parse_dates)
    tst_nl = pd.read_csv(os.path.join(dp.output_path, 'tst_nl_merged.csv'), parse_dates = parse_dates)

    tst_nl = tst_nl.rename({
        '성별': 'Gender', 
        '나이':'Age',
        '혈압_수축기': 'SBP', 
        '체온':'BT',
        '맥박':'HR', 
        '호흡':'RR', 
        '혈액검사시점' : 'blood_time',
        'WBC count' : 'WBC Count'}, 
        axis=1)
    tst_nl = tst_nl.drop(['datetime'], axis = 1)

    tst_abn = tst_abn.rename({
        '성별': 'Gender', 
        '나이':'Age',
        '혈압_수축기': 'SBP', 
        '체온':'BT',
        '맥박':'HR', 
        '호흡':'RR', 
        '혈액검사시점' : 'blood_time',
        'WBC count' : 'WBC Count'}, 
        axis=1)
    tst_abn = tst_abn.drop(['event','detection', 'datetime'], axis = 1)

    set1 = tst_abn.columns
    set2 = tst_nl.columns 
    set1.difference(set2)

    tst_abn['is_abn'] = 1
    tst_nl['is_abn'] = 0

    whole_train = abn_nl_concat(trn_abn, trn_nl)
    whole_train = whole_train.drop(['type'], axis = 1)
    
    whole_test = abn_nl_concat(tst_abn, tst_nl)
    whole_test = whole_test.drop(['type', '대체번호.1', '생년월일', '원자료번호'], axis = 1)

    # Filter chem and cbc
    whole_train = adjust_cbc(whole_train)
    whole_train = adjust_chem(whole_train)

    whole_test = adjust_cbc(whole_test)
    whole_test = adjust_chem(whole_test)


    var_list = list(set(list(whole_test.columns))
                    - set([ 'Patient', 'adjusted_time', 'detection_date', 'detection_time', 'event_date', 'event_time', 'blood_time', 'is_abn', 'target', '구분', '대체번호']) )
    
    print('var list: ', var_list)

    window_len = 8

    sequence_data_train = make_sequence_data(whole_train, window_len = window_len, var_list = var_list, index = 'Patient')
    sequence_data_test = make_sequence_data(whole_test, window_len = window_len, var_list = var_list, index = 'Patient')

    path = os.path.join(dp.output_path, 'DAT01_train.pickle')
    with open(path, 'wb') as f:
        pickle.dump(sequence_data_train, f)

    path = os.path.join(dp.output_path, 'DAT01_test.pickle')
    with open(path, 'wb') as f:
        pickle.dump(sequence_data_test, f)

    return sequence_data_train, sequence_data_test, var_list 

sequence_data_train, sequence_data_test = make_sequence_train()

train_data = make_2d_data(
    sequence_data_train, 
    var_list = var_list, 
    output_path = dp.output_path, 
    output_file = 'train_final.csv')

test_data = make_2d_data(
    sequence_data_test, 
    var_list = var_list, 
    output_path = dp.output_path, 
    output_file = 'test_final.csv')


