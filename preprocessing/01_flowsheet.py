#! /home/sooheang/anaconda3/bin/python

import os 
import re 
import pandas as pd
import numpy as np 
import warnings 
import seaborn as sns 
import datetime as dt

from datetime import datetime
from typing import Dict, Optional 
from dfply import * 

DATA_PATH = '/data/datasets/rrs-data/10yrs_raw_data'
OUTPUT_PATH = '/data/datasets/rrs-data/10yrs_refined_data'

#####
# Load dataset 
trn_abn_flow = pd.read_csv(os.path.join(DATA_PATH, '시험군_TPR_FLOWSHEET.csv'), \
  encoding = 'cp949', dtype=object)
tst_abn_flow = pd.read_csv(os.path.join(DATA_PATH, 'TEST시험군_TPR_FLOWSHEET.csv'), \
  encoding = 'cp949', dtype=object)
trn_nl_flow = pd.read_csv(os.path.join(DATA_PATH, '대조군_TPR_FLOWSHEET.csv'), \
  encoding = 'cp949', dtype=object)
tst_nl_flow = pd.read_csv(os.path.join(DATA_PATH, 'TEST대조군_TPR_FLOWSHHET.csv'), \
  encoding = 'cp949', dtype=object)

#####
# user=defined functions 
def make_patient_index(df):
  df['patient'] = df['대체번호'] + '_' + df['입원일자']
  return df

def check_numeric(x):
  try:
    good = float(x)
    return np.nan
  except:
    return x

def var_preprocess(x):
  if x in ['Respiratory Rate', 'Respiratory  rate', 'Respiratory rate']:
    return 'RR'
  elif x in ['SaO2/Pulse Ox(%)', 'SaO2/PO2(A)', 'Sao2/pulse Ox(%)', 'SaO2',
              'SaO2/pulse Ox(%)', 'SaO2/Pulse', 'SaO2/Pulse Ox (%)', 'SaO2/pulse(%)']:
    return 'SaO2'
  elif x in ['Blood Pressure Cuff/NIBP', 'Blood Pressure A-line', 'Blood Pressure NBP', 'Blood Pressure NIBP',
              'Blood Pressure Art Line', 'Blood Pressure C10429uff/NIBP', 'Blood pressure(Lt.radial)']:
    return 'SBP'
  elif x in ['Temperature']:
    return 'Temp'
  elif x in ['Heart Rate']:
    return 'HR'

def to_numeric(x):
  try:
    return float(x)
  except:
    num = re.findall('[-+]?\d*\.\d+|\d+', x)
    try:
      return float(num[0])
    except:
      return np.nan      

def make_pivot(dataset, abn = True):
  dataset['측정결과'] = dataset['측정결과'].apply(lambda x: to_numeric(x))
  if abn:
    dataset_pivot = dataset.pivot_table(index = ['patient', 'CPR_기도삽관_처방일자', 'CPR_기도삽관_처방시간',
                                                  '측정일자', '측정시간'],
                                        columns = ['항목'],
                                        values = ['측정결과'])
    dataset_pivot = dataset_pivot.reset_index(col_level = 1)
    dataset_pivot.columns = dataset_pivot.columns.droplevel()
    return dataset_pivot
  else:
    dataset_pivot = dataset.pivot_table(index = ['patient', '측정일자', '측정시간'],
                                        columns = ['항목'],
                                        values = ['측정결과'])
    dataset_pivot = dataset_pivot.reset_index(col_level = 1)
    dataset_pivot.columns = dataset_pivot.columns.droplevel()
    return dataset_pivot

def make_datetime(date, time):
  try:
    datetime_string = str(date)[:8] + ' ' + str(time)[:4]
    datetime_object = datetime.strptime(datetime_string, "%Y%m%d %H%M")
    return datetime_object
  except:
    datetime_string = str(date)[:8]
    datetime_object = datetime.strptime(datetime_string, '%Y%m%d')
    hour_delta = pd.to_timedelta(int(str(time)[:2]), unit = 'h')
    minute_delta = pd.to_timedelta(int(str(time)[2:4]), unit = 'm')
    return datetime_object + hour_delta + minute_delta

def drop_multi_event(data):
  patient_list = pd.unique(data['patient'])
  new_data = pd.DataFrame()
  for idx, patient in enumerate(patient_list):
    one_patient_data = data[data['patient'] == patient]
    event_record = pd.unique(one_patient_data['event_time'])
    first_event = min(event_record)
    one_patient_data['event_time'] = first_event
    new_data = new_data.append(one_patient_data)
  return new_data

def drop_after_event(df):
  df['time'] = df['event_time'] - df['measurement_time']
  df = df.drop(df[df['time'] <= pd.to_timedelta(0)].index)
  del df['time']
  df = df.reset_index(drop=True)
  return df

def get_resampled(df, index = 'patient', time = 'adjusted_time', freq = '1H'):
  df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill()))
  df = df.drop([index], axis=1)
  df['SaO2'] = df['SaO2'].fillna(95)
  df = df.dropna()
  df = df.reset_index()
  return df

########
# show sample dataset
trn_abn_flow >> sample(frac=0.0001, replace=False)
tst_abn_flow >> sample(frac=0.0001, replace=False)
trn_nl_flow >> sample(frac=0.0001, replace=False)
tst_nl_flow >> sample(frac=0.0001, replace=False)

#trn_abn_flow.head(100)

## Pre-processing 
dataset_list = [trn_abn_flow, trn_nl_flow, tst_abn_flow, tst_nl_flow]
dataset_dict = {'trn_abn' : trn_abn_flow, 'trn_nl' : trn_nl_flow,
                'tst_abn' : tst_abn_flow, 'tst_nl' : tst_nl_flow}

for data in dataset_list:
  data['항목'] = data['항목'].apply(lambda x: var_preprocess(x))
  data['이상값'] = data['측정결과'].apply(lambda x: check_numeric(x))

for name, data in dataset_dict.items():
  weird_data = data.iloc[data['이상값'].dropna().index]
  weird_data.to_csv(os.path.join(OUTPUT_PATH, f'{name}_flowsheet_weird.csv'), encoding = 'cp949', index = False)

## Generate patient index 

trn_abn_flow = make_patient_index(trn_abn_flow)
trn_nl_flow = make_patient_index(trn_nl_flow)
tst_abn_flow = make_patient_index(tst_abn_flow)
tst_nl_flow = make_patient_index(tst_nl_flow)

trn_abn_pivot = make_pivot(trn_abn_flow, abn=True)
trn_nl_pivot = make_pivot(trn_nl_flow, abn=False)
tst_abn_pivot = make_pivot(tst_abn_flow, abn=True)
tst_nl_pivot = make_pivot(tst_nl_flow, abn=False)


## eventtime 
pivot_dict = {'trn_abn': trn_abn_pivot,
              'trn_nl' : trn_nl_pivot,
              'tst_abn' : tst_abn_pivot,
              'tst_nl' : tst_nl_pivot}

for i in trn_abn_pivot['CPR_기도삽관_처방시간']:
  if len(str(i)) != 4:
    print(i)
  else:
    pass


trn_abn_pivot['event_time'] = trn_abn_pivot.apply(lambda x: make_datetime(x['CPR_기도삽관_처방일자'], x['CPR_기도삽관_처방시간']),
                                                  axis = 1)
tst_abn_pivot['event_time'] = tst_abn_pivot.apply(lambda x: make_datetime(x['CPR_기도삽관_처방일자'], x['CPR_기도삽관_처방시간']),
                                                  axis = 1)

for name, data in pivot_dict.items():
    print(name)
    data['measurement_time'] = data.apply(lambda x: make_datetime(x['측정일자'], x['측정시간']), axis = 1)

########
# delete unused feature set
trn_abn_pivot = trn_abn_pivot.drop(columns = ['CPR_기도삽관_처방일자', 'CPR_기도삽관_처방시간', '측정일자', '측정시간'])
trn_nl_pivot = trn_nl_pivot.drop(columns = ['측정일자', '측정시간'])
tst_abn_pivot = tst_abn_pivot.drop(columns = ['CPR_기도삽관_처방일자', 'CPR_기도삽관_처방시간', '측정일자', '측정시간'])
tst_nl_pivot = tst_nl_pivot.drop(columns = ['측정일자', '측정시간'])


#######
# idenfity patients who suffer from multiple events
source = []
patient = []
count = []
event_time = []
for name, data in pivot_dict.items():
  try:
    multiple_cpr_count = 0
    print('-----', name, '-----')
    patient_list = pd.unique(data['patient'])
    for patient_no in patient_list:
      one_patient_data = data[data['patient'] == patient_no]
      cpr_event_time = pd.unique(one_patient_data['event_time']) 
      cpr_count = len(cpr_event_time)
      if cpr_count != 1:
        multiple_cpr_count += 1
        source.append(name)
        patient.append(patient_no)
        count.append(cpr_count)
        event_time.append(cpr_event_time)
    print(f'{len(patient_list)}명 중 {multiple_cpr_count}명이 복수 개의 event time 보유.')
  except:
    pass

multi_event_data = pd.DataFrame({'원천' : source,
                                 'patient' : patient,
                                 'Event 발생 횟수' : count,
                                 'Event Time' : event_time})

multi_event_data.to_csv(os.path.join(OUTPUT_PATH,'Multi_event.csv'), 
  encoding = 'cp949')

trn_abn_test = drop_multi_event(trn_abn_pivot)
tst_abn_test = drop_multi_event(tst_abn_pivot)

patient_list = pd.unique(tst_abn_test['patient'])

for patient_no in patient_list:
  one_patient_data = tst_abn_test[tst_abn_test['patient'] == patient_no]
  cpr_event_time = pd.unique(one_patient_data['event_time']) 
  cpr_count = len(cpr_event_time)
  if cpr_count != 1:
    print(patient_no)

trn_abn_test = drop_after_event(trn_abn_test)
tst_abn_test = drop_after_event(tst_abn_test)

#######
## resampling 
trn_abn_test['adjusted_time'] = trn_abn_test['measurement_time'].dt.round('H')
trn_nl_pivot['adjusted_time'] = trn_nl_pivot['measurement_time'].dt.round('H')
tst_abn_test['adjusted_time'] = tst_abn_test['measurement_time'].dt.round('H')
tst_nl_pivot['adjusted_time'] = tst_nl_pivot['measurement_time'].dt.round('H')

trn_abn_resampled = get_resampled(trn_abn_test)
trn_nl_resampled = get_resampled(trn_nl_pivot)
tst_abn_resampled = get_resampled(tst_abn_test)
tst_nl_resampled = get_resampled(tst_nl_pivot)

resample_dict = {'trn_abn' : trn_abn_resampled,
                 'trn_nl' : trn_nl_resampled,
                 'tst_abn' : tst_abn_resampled,
                 'tst_nl' : tst_nl_resampled}

for name, data in resample_dict.items():
    print(name)
    data.to_csv(os.path.join(OUTPUT_PATH, f'{name}_flowsheet.csv'), index = False, encoding = 'cp949')
      