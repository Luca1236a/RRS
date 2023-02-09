#! /home/sooheang/anaconda3/bin/python

import os 
import re 
import pandas as pd
import numpy as np 
import warnings 
import seaborn as sns 
import datetime as dt
import random 

from datetime import datetime
from typing import Dict, Optional 

DATA_PATH = '/data/datasets/rrs-data/10yrs_refined_data'
OUTPUT_PATH = '/data/datasets/rrs-data/10yrs_samples'

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

def get_patient_list(data_path: str, data_file: str, out_path: str, out_file: str, meta_file: str, verbose = True):
  df = pd.read_csv(os.path.join(data_path, data_file), \
    encoding = 'cp949', dtype=object)

  df = df. \
    assign(patient_id = lambda x: x.patient.str.slice(stop=8))
  
  df['adjusted_time'] = df['adjusted_time'].\
    astype('datetime64[ns]')

  dfm = pd.read_csv(os.path.join(data_path, meta_file), encoding = 'CP949')
  
  df = df.merge(dfm,
    left_on = ['patient_id'],
    right_on = ['patient_id'],
    how = 'left'
  )

  dfp = df.\
    sort_values('adjusted_time').\
    drop_duplicates(subset = 'patient_id', keep = 'first')

  dfp = dfp[dfp['adjusted_time'].\
        between(left = '2010-01-01', right = '2016-12-31')]
  
  if verbose:
    dfp.groupby('adjusted_time').size()    
    print('Number of 2010-2016 event patients:', dfp.shape[0])

  if dfp.shape[0] > 100:
    dfp = dfp.sample(n=100, random_state=1)

  dfp[['patient_id', 'gender', 'birthday', 'event_time']].\
    to_csv(os.path.join(out_path, out_file), index = False)

  return dfp 

def main(verbose = True): 

  if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

  # Gender and birthday data
  trn_abn_gender = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_GENDER_DATA), encoding = 'CP949')
  trn_abn_gender = trn_abn_gender.assign(data_type = 'train')
  trn_abn_gender = trn_abn_gender.assign(abn = True)

  trn_nl_gender = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_GENDER_DATA), encoding = 'CP949')
  trn_nl_gender = trn_nl_gender.assign(abn = False)
  trn_nl_gender = trn_nl_gender.assign(data_type = 'train')

  tst_abn_gender = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_GENDER_DATA), encoding = 'CP949')
  tst_abn_gender = tst_abn_gender.assign(data_type = 'test')
  tst_abn_gender = tst_abn_gender.assign(abn = True)

  tst_nl_gender = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_GENDER_DATA), encoding = 'CP949')
  tst_nl_gender = tst_nl_gender.assign(data_type = 'test')
  tst_nl_gender = tst_nl_gender.assign(abn = False)

  trn_gender = pd.concat([trn_abn_gender, trn_nl_gender])
  trn_gender = trn_gender.drop_duplicates(subset = 'patient_id', keep = 'first')

  tst_gender = pd.concat([tst_abn_gender, tst_nl_gender])
  tst_gender = tst_gender.drop_duplicates(subset = 'patient_id', keep = 'first')

  meta = pd.concat([trn_gender, tst_gender])  
  if verbose:
    meta.groupby('data_type').size()
    meta.groupby('abn').size()
    meta.groupby(['data_type','abn']).size()

  trn_abn_blood = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_BLOOD_DATA), encoding='cp949')
  trn_abn_blood = trn_abn_blood.assign(data_type = 'train')
  trn_abn_blood = trn_abn_blood.assign(abn = True)

  trn_nl_blood = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_BLOOD_DATA), encoding='cp949')
  trn_nl_blood = trn_nl_blood.assign(data_type = 'train')
  trn_nl_blood = trn_nl_blood.assign(abn = False)

  tst_abn_blood = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_BLOOD_DATA), encoding='cp949')
  tst_abn_blood = tst_abn_blood.assign(data_type = 'test')
  tst_abn_blood = tst_abn_blood.assign(abn = True)

  tst_nl_blood = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_BLOOD_DATA), encoding='cp949')
  tst_nl_blood = tst_nl_blood.assign(data_type = 'test')
  tst_nl_blood = tst_nl_blood.assign(abn = False)

  trn_blood = pd.concat([trn_abn_blood, trn_nl_blood])
  tst_blood = pd.concat([tst_abn_blood, tst_nl_blood])
  blood = pd.concat([trn_blood, tst_blood])  
  
  blood = blood. \
    assign(patient_id = lambda x: x.patient.str.slice(stop=8))

  # drop null measurement_time
  blood = blood.dropna(subset = ['measurement_time'])  

  blood = blood. \
    drop_duplicates(subset = 'patient_id', keep = 'first')
  blood = blood.drop(['patient', 'adjusted_time'], axis = 1)
  if verbose: 
    blood[blood.pipe(lambda x: x['patient_id'] == 'A027IXGS')]
  
  df = meta.merge(blood,
    left_on = ['patient_id', 'data_type', 'abn'],
    right_on = ['patient_id', 'data_type', 'abn'],
    how = 'left'
  )

  trn_abn_lab = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_LAB_DATA), encoding='cp949')
  trn_abn_lab = trn_abn_lab.assign(data_type = 'train')
  trn_abn_lab = trn_abn_lab.assign(abn = True)

  trn_nl_lab = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_LAB_DATA), encoding='cp949') 
  trn_nl_lab = trn_nl_lab.assign(data_type = 'train')
  trn_nl_lab = trn_nl_lab.assign(abn = False)

  tst_abn_lab = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_LAB_DATA), encoding='cp949')
  tst_abn_lab = tst_abn_lab.assign(data_type = 'test')
  tst_abn_lab = tst_abn_lab.assign(abn = True)
  
  tst_nl_lab = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_LAB_DATA), encoding='cp949') 
  tst_nl_lab = tst_nl_lab.assign(data_type = 'test')
  tst_nl_lab = tst_nl_lab.assign(abn = False)

  trn_lab = pd.concat([trn_abn_lab, trn_nl_lab])
  tst_lab = pd.concat([tst_abn_lab, tst_nl_lab])
  lab = pd.concat([trn_lab, tst_lab])  
  
  lab = lab. \
    assign(patient_id = lambda x: x.patient.str.slice(stop=8))
  
  # drop null measurement_time
  lab = lab.dropna(subset = ['measurement_time'])  

  lab = lab. \
    drop_duplicates(subset = 'patient_id', keep = 'first')
  lab = lab.drop(['patient', 'adjusted_time', 'event_time', 'measurement_time'], axis = 1)
  if verbose: 
    lab[lab.pipe(lambda x: x['patient_id'] == 'A027IXGS')]
  
  df = df.merge(lab,
    left_on = ['patient_id', 'data_type', 'abn'],
    right_on = ['patient_id', 'data_type', 'abn'],
    how = 'left'
  )
  df.columns

  trn_abn_vital = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_VITAL_DATA), encoding='cp949')
  trn_abn_vital = trn_abn_vital.assign(data_type = 'train')
  trn_abn_vital = trn_abn_vital.assign(abn = True)

  trn_nl_vital = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_VITAL_DATA), encoding='cp949') 
  trn_nl_vital = trn_nl_vital.assign(data_type = 'train')
  trn_nl_vital = trn_nl_vital.assign(abn = False)

  tst_abn_vital = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_VITAL_DATA), encoding='cp949')
  tst_abn_vital = tst_abn_vital.assign(data_type = 'test')
  tst_abn_vital = tst_abn_vital.assign(abn = True)

  tst_nl_vital = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_VITAL_DATA), encoding='cp949') 
  tst_nl_vital = tst_nl_vital.assign(data_type = 'test')
  tst_nl_vital = tst_nl_vital.assign(abn = False)

  tst_vital = pd.concat([tst_abn_vital, tst_nl_vital])
  trn_vital = pd.concat([trn_abn_vital, trn_nl_vital])
  vital = pd.concat([trn_vital, tst_vital]) 
  
  vital = vital. \
    assign(patient_id = lambda x: x.patient.str.slice(stop=8))
  
  if verbose:
    vital[vital.pipe(lambda x: x['patient_id'] == 'A027IXGS')]
    vital[vital.pipe(lambda x: x['patient_id'] == 'A065SHX6')]
  
  # drop null measurement_time
  vital = vital.dropna(subset = ['measurement_time'])  

  vital = vital. \
    drop_duplicates(subset = 'patient_id', keep = 'first')
  vital = vital.drop(['patient', 'adjusted_time', 'event_time', 'measurement_time'], axis = 1)
  if verbose:
    vital[vital.pipe(lambda x: x['patient_id'] == 'A027IXGS')]
    vital[vital.pipe(lambda x: x['patient_id'] == 'A065SHX6')]
  
  df = df.merge(vital,
    left_on = ['patient_id', 'data_type', 'abn'],
    right_on = ['patient_id', 'data_type', 'abn'],
    how = 'left'
  )
  df

  #####
  # Select ABN patient list on cnuhh
  is_train = df['data_type'] == 'train'
  is_abn = df['abn'] == True
  is_date = pd.to_datetime(df['measurement_time']) < '2015-12-31 23:59:59'
  is_hr = df['HR'].notnull()  
  is_alt = df['ALT'].notnull()

  if verbose: 
    print('before 2017')
    df[is_date]
    print('train dataset')
    df[is_train]
    print('Abnormal dataset')
    df[is_abn]

  # Select train abnormal sample 
  trn_a = df[is_train & is_abn & is_date & is_hr & is_alt]
  trn_b = df[is_train & is_abn & is_date & is_hr & ~is_alt].sample(n=300, random_state=1) #112
  pd.concat([trn_a, trn_b]).to_csv(os.path.join(OUTPUT_PATH, '10yrs_trn_abn_sample.csv'), index = False)

  #trn_a = df[is_train & ~is_abn & is_date & is_hr & is_alt]
  #trn_b = df[is_train & ~is_abn & is_date & is_hr & ~is_alt]
  #trn_c = df[is_train & ~is_abn & is_date & ~is_hr & ~is_alt].sample(n=404, random_state=1)
  #pd.concat([trn_a, trn_b, trn_c]).to_csv(os.path.join(OUTPUT_PATH, '10yrs_trn_nl_sample.csv'), index = False)
  
  tst_a = df[~is_train & is_abn & is_date & is_hr & is_alt]
  tst_b = df[~is_train & is_abn & is_date & is_hr & ~is_alt].sample(n=200, random_state=1)
  pd.concat([tst_a, tst_b]).to_csv(os.path.join(OUTPUT_PATH, '10yrs_tst_abn_sample.csv'), index = False)

  #tst_a = df[~is_train & ~is_abn & is_date & is_hr & is_alt]
  #tst_b = df[~is_train & ~is_abn & is_date & is_hr & ~is_alt].sample(n=492, random_state=1)
  #pd.concat([tst_a, tst_b]).to_csv(os.path.join(OUTPUT_PATH, '10yrs_tst_nl_sample.csv'), index = False)
  
if __name__ == '__main__':
  main(verbose = True)
