#! /usr/bin/python 

from absl import app, flags

import os 
import re 
import pandas as pd
import numpy as np 
import warnings 
import seaborn as sns 
import datetime as dt
import random 

from datetime import datetime
from typing import Dict, Optional, List  

from DataClass import DataPath, VarSet, ModelSet

def main(DataPath, verbose: bool = True) -> None:

  if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


  trn_abn_sign = pd.read_csv( 
    os.path.join(DATA_PATH, TRAIN_ABN_SIGN_DATA),
    sep = ','
    )

  trn_nl_sign = pd.read_excel(
    io = os.path.join(DATA_PATH, TRAIN_NL_SIGN_DATA),    
    coding = 'CP949'
  )

  # Get patinet ID
  trn_abn_patient = trn_abn_sign['대체번호'].drop_duplicates()
  trn_nl_patient = trn_nl_sign['원자료번호'].drop_duplicates()

  if verbose:
    print('훈련 데이터 총 환자 수: ', len(trn_abn_patient) + len(trn_nl_patient))
    print('   - ABN 환자 수   : ', len(trn_abn_patient))
    print('   - NL 환자 수    : ', len(trn_nl_patient))


  # drop variables
  abn_drop_list = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '원자료번호','Unnamed: 12', '혈압_이완기']
  nl_drop_list = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수']
  event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']

  trn_abn = trn_abn.drop(abn_drop_list, axis = 1)
  trn_nl = trn_nl.drop(nl_drop_list, axis = 1)
  trn_event = trn_event.drop(event_drop_list, axis = 1)

  trn_abn_blood = pd.read_excel( 
    io = os.path.join(DATA_PATH, BLOOD_DATA),
    sheet_name=BLOOD_TRAIN_ABN_SHEET,
    coding = 'CP949'
    )
  
  if verbose:
    tmp = trn_abn_blood.groupby('대체번호').count()
    tmp
    tmp.to_excel(\
      os.path.join(OUTPUT_PATH, 'patients_lab_counts.xlsx')\
    )
    for i in trn_abn_blood.columns[13:]:
      print('value count:', i)
      print(tmp[13:].value_counts())
      tmp[trn_abn_blood.columns[13:]].apply(pd.Series.value_counts).to_excel(\
        os.path.join(OUTPUT_PATH, 'trn_abn_blood_value_counts.xlsx'))

  return pass 


def drop_row(df, drop_list):
  df = df.drop(drop_list, axis=0)
  df = df.reset_index(drop=True)
  return df

def weird_time(df):
  drop_index = []
  for i in range(len(df)):
    if len(df['측정시각'].iloc[i]) != 4:
      print(i,'th row is weird.', '-->', df['측정시각'].iloc[i])
      drop_index.append(i)
  return drop_index
  
def main(verbose = True):
  
  # Load event dataset 
  event_trn = pd.read_excel(
    os.path.join(DATA_PATH, EVENT_DATA),
    sheet_name = EVENT_SHEET_ABN_TRN,
    usecols= 'E,F'
  )
  event_trn = event_trn.rename(columns = {
      '대체번호': 'patient_id',
      '종류': 'event_type'
    }
  )

  if verbose:
    print('Number of event types in train data:', event_trn.groupby('event_type').size())

  event_tst = pd.read_excel(
    os.path.join(DATA_PATH, EVENT_DATA),
    sheet_name = EVENT_SHEET_ABN_TST
  )
  event_tst = event_tst.rename(columns = {
    '대체번호': 'patient_id',
    '종류': 'event_type'
  })  
  if verbose:
    print('Number of event types in test data:', event_tst.groupby('event_type').size())
    print('where  0: event 생존')
    print('       1: event 사망')
    print('       2: event없이 사망')
    
  # Load Vital sign dataset 
  vital_trn = pd.read_excel(
    os.path.join(DATA_PATH, VITAL_TRAIN),
    sheet_name = '', 
    usecols = ''
  )
  # Add event information column asdf
  
  
  # Load vital sign dataset 
  trn_abn = pd.read_excel(
    os.path.join(DATA_PATH, '화순abn_ward(외부).xlsx'), 
    sheet_name='RRTabn', 
    dtype={'측정시각':str, '측정일' : str}
    )
    
  trn_event = pd.read_excel('Raw/TRN01-2. data_event_train.xlsx')
  trn_abn_blood = pd.read_csv('Raw/TRN01-3. data_abn_blood_train.csv')
  trn_nl = pd.read_csv('Raw/TRN02-1. data_nl_train.csv', dtype={'측정시각':str, '측정일' : str}, encoding='CP949')
  trn_exclude = pd.read_csv('Raw/TRN02-2. data_exclude_train.csv', encoding='CP949', header=None, names=['Name', ''])
  trn_nl_blood = pd.read_csv('Raw/TRN02-3. data_nl_blood_train.csv')


if __name__=="__main__": 
  main() 