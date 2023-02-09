#! /usr/bin/python 

from operator import index
import os 
import numpy as np 
import pandas as pd 
import datetime as dt 

DATA_PATH = '/data/datasets/rrs-data/10yrs_refined_data'
RAW_PATH = '/data/datasets/rrs-data/10yrs_raw_data'

TRAIN_ABN_GENDER_DATA = 'trn_abn_gender.csv'
TRAIN_ABN_BLOOD_DATA = 'trn_abn_cbc.csv'
TRAIN_ABN_LAB_DATA = 'trn_abn_chem.csv'
TRAIN_ABN_VITAL_DATA = 'trn_abn_flowsheet.csv'
TRAIN_ABN_VITAL_ICU_DATA = 'trn_abn_obsrv.csv' 

TRAIN_NL_GENDER_DATA = 'trn_nl_gender.csv'
TRAIN_NL_BLOOD_DATA = 'trn_nl_cbc.csv'
TRAIN_NL_LAB_DATA = 'trn_nl_chem.csv'
TRAIN_NL_VITAL_DATA = 'trn_nl_flowsheet.csv'
TRAIN_NL_VITAL_ICU_DATA = 'trn_nl_obsrv.csv' 

TEST_ABN_GENDER_DATA = 'tst_abn_gender.csv'
TEST_ABN_BLOOD_DATA = 'tst_abn_cbc.csv'
TEST_ABN_LAB_DATA = 'tst_abn_chem.csv'
TEST_ABN_VITAL_DATA = 'tst_abn_flowsheet.csv'
TEST_ABN_VITAL_ICU_DATA = 'tst_abn_obsrv.csv' 

TEST_NL_GENDER_DATA = 'tst_nl_gender.csv'
TEST_NL_BLOOD_DATA = 'tst_nl_cbc.csv'
TEST_NL_LAB_DATA = 'tst_nl_chem.csv'
TEST_NL_VITAL_DATA = 'tst_nl_flowsheet.csv'
TEST_NL_VITAL_ICU_DATA = 'tst_nl_obsrv.csv' 
"""
target labeling
1. abn일 경우
- event time보다 측정시간이 나중일 경우 데이터 제거
- time이 detection 타임과 event 타임 사이일 경우 taget = 1
2. nl일 경우
- 모두 0
"""

def get_target_revised(time, detection, event):
    if detection <= time <= event:
        return 1
    else:
        return 0
    
def get_target_df(df, is_abn = True):
  if is_abn:
    df['target'] = df['event_time'] - df['adjusted_time']
    df = df.drop(df[df['target'] <= pd.to_timedelta(0)].index)
    df['target'] = df.apply(lambda x: get_target_revised(x['adjusted_time'], x['detection'],
                                                    x['event_time']), axis=1)
    return df
  else:
    df['target'] = 0
    return df 

def get_resampled(df, index, time = 'adjusted_time', freq = '1H', without_bfill = True):
  if without_bfill:
    df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill()))
    df = df.drop([index], axis=1)
    df = df.dropna()
    df = df.reset_index(level=[1]).reset_index()
  else:
    df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill().bfill()))
    df = df.drop([index], axis=1)
    df = df.dropna()
    df = df.reset_index(level=[1]).reset_index()       
  return df
  
def get_merge_data(df, df_bld):
  df_bld = get_resampled(df_bld, 'patient')
  df_merged = df.merge(df_bld,
                        left_on = ['patient', 'adjusted_time'],
                        right_on = ['patient', 'adjusted_time'],
                        how = 'left')
  df_merged = get_resampled(df_merged, 'patient', without_bfill=False)
  return df_merged

def main(verbose: str = True):

  # Build train dataset

  # Gender and birthday data
  trn_abn_gender = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_GENDER_DATA), encoding = 'CP949')
  trn_abn_gender = trn_abn_gender.assign(data_type = 'train')
  trn_abn_gender = trn_abn_gender.assign(abn = True)
  
  trn_nl_gender = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_GENDER_DATA), encoding = 'CP949')
  trn_nl_gender = trn_nl_gender.assign(data_type = 'train')
  trn_nl_gender = trn_nl_gender.assign(abn = False)

  trn_gender = pd.concat([trn_abn_gender, trn_nl_gender])
  if verbose:
    #len(trn_abn_gender['patient_id'])-len(trn_abn_gender['patient_id'].drop_duplicates())
    #len(trn_nl_gender['patient_id'])-len(trn_nl_gender['patient_id'].drop_duplicates())
    print('duplicated patient ids in training dataset:', \
      len(trn_gender['patient_id'])-len(trn_gender['patient_id'].drop_duplicates()))
    #trn_gender[trn_gender.patient_id.duplicated()].to_csv('abn_nl_duplicated_id.csv', index=False)
    #trn_gender[trn_gender.pipe(lambda x: x.patient_id == 'AG7KY55L')]
    trn_gender.groupby('data_type').size()
    trn_gender.groupby('abn').size()
    trn_gender.groupby('patient_id').size()

  #####
  # NOTICE
  # RUN: drop duplicated ids 
  trn_gender = trn_gender.drop_duplicates(subset = 'patient_id', keep='first')
  
  # WBC count, Hgb, and Platelet Count 
  trn_abn_blood = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_BLOOD_DATA), encoding='cp949')
  
  #trn_abn_blood.pivot_table(
  #  index = ['patient'],
  #  columns=['adjusted_time'],
  #  values= ['Hgb', 'Platelet Count', 'WBC Count']
  #).reset_index()

  #trn_abn_blood = trn_abn_blood.assign(target = 0)
  #(pd.to_datetime(trn_abn_blood['event_time'].dtypes) - pd.to_datetime(trn_abn_blood['measurement_time']))
  #trn_abn_blood.apply(lambda x: (X.adjusted_time - X.event_time).dt.days, axis = 0)
  trn_abn_blood = trn_abn_blood.assign(data_type = 'train')
  trn_abn_blood = trn_abn_blood.assign(abn = True)

  trn_nl_blood = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_BLOOD_DATA), encoding='cp949')
  trn_nl_blood = trn_nl_blood.assign(data_type = 'train')
  trn_nl_blood = trn_nl_blood.assign(abn = False)

  trn_blood = pd.concat([trn_abn_blood, trn_nl_blood])
  if verbose:
    trn_blood.describe()
    trn_blood.groupby('abn').size()
    trn_blood.groupby('patient').size()
  
  trn_abn_lab = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_LAB_DATA), encoding='cp949')
  trn_abn_lab = trn_abn_lab.assign(data_type = 'train')
  trn_abn_lab = trn_abn_lab.assign(abn = True)

  trn_nl_lab = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_LAB_DATA), encoding='cp949') 
  trn_nl_lab = trn_nl_lab.assign(data_type = 'train')
  trn_nl_lab = trn_nl_lab.assign(abn = False)

  trn_lab = pd.concat([trn_abn_lab, trn_nl_lab])
  if verbose:
    trn_lab.groupby('abn').size()
    trn_lab.groupby('patient').size()

  trn_abn_vital = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_VITAL_DATA), encoding='cp949')
  trn_abn_vital = trn_abn_vital.assign(data_type = 'train')
  trn_abn_vital = trn_abn_vital.assign(abn = True)
  
  trn_nl_vital = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_VITAL_DATA), encoding='cp949') 
  trn_nl_vital = trn_nl_vital.assign(data_type = 'train')
  trn_nl_vital = trn_nl_vital.assign(abn = False)

  trn_vital = pd.concat([trn_abn_vital, trn_nl_vital])
  
  if verbose:
    trn_vital.groupby('abn').size()
    trn_vital.groupby('patient').size()

  trn_abn_vital_icu = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_VITAL_ICU_DATA), encoding='cp949')

  trn_blood = trn_blood.assign(patient_id = lambda x: x.patient.str.slice(stop=8))
  
  if verbose: 
    trn_gender.groupby('patient_id').size() 
    trn_blood.groupby('patient_id').size() 

  trn_abn = trn_abn_gender.merge(trn_abn_blood, 
                        left_on = ['patient_id'],
                        right_on = ['patient_id'],
                        how = 'right'
                      )

  trn_abn = trn_abn.merge(trn_abn_lab, \
    left_on = ['patient', 'adjusted_time'],
    right_on = ['patient', 'adjusted_time'],
    how = 'outer'
    )
  
  test[test.pipe(lambda x: x['patient_id'] == 'A027IXGS')]

  trn_abn_gender.head()
  trn_abn_vital.columns 
  trn_abn_vital_icu.columns 
  
  trn_abn = trn_abn_vital.merge(trn_abn_vital_icu, \
    left_on = ['patient', 'adjusted_time'], \
    right_on = ['patient', 'adjusted_time'])


  trn_nl_blood = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_BLOOD_DATA), encoding='cp949')
  trn_nl_lab = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_LAB_DATA), encoding='cp949')
  trn_nl_vital = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_VITAL_DATA), encoding='cp949')
  trn_nl_vital_icu = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_VITAL_ICU_DATA), encoding='cp949')

  tst_abn_blood = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_BLOOD_DATA), encoding='cp949')
  tst_abn_lab = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_LAB_DATA), encoding='cp949')
  tst_abn_vital = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_VITAL_DATA), encoding='cp949')
  tst_abn_vital_icu = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_VITAL_ICU_DATA), encoding='cp949')

  tst_nl_blood = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_BLOOD_DATA), encoding='cp949')
  tst_nl_lab = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_LAB_DATA), encoding='cp949')
  tst_nl_vital = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_VITAL_DATA), encoding='cp949')
  tst_nl_vital_icu = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_VITAL_ICU_DATA), encoding='cp949')

  # merge dataset 





if __name__ == '__main__':
  main()