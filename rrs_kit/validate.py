import os 
import numpy as np 
import pandas as pd 
import pickle 

from DataClass import DataPath
from utils import (
  filter_sign,
  adjust_cbc,
  adjust_chem,
  get_target_df,
  get_merge_data,
  make_timestamp,
  make_sequence_data,
  make_2d_data
)

def get_event_id(verbose = True) -> pd.DataFrame:

  dp = DataPath()

  valid_abn_trn = pd.read_excel(
    os.path.join(dp.valid_path, dp.valid_abn_event[0])
  )

  valid_abn_tst = pd.read_csv(
    os.path.join(dp.valid_path, dp.valid_abn_event[1]),
    sep = ',',
    encoding= ' CP949'
  )
  valid_abn_tst['detection_time'] = valid_abn_tst[['detection ']]
  valid_abn_tst = valid_abn_tst.drop(['detection '], axis = 1)

  valid_abn = pd.concat([valid_abn_trn, valid_abn_tst])
  #set(valid_abn_trn.columns).difference(valid_abn_tst.columns)  
  
  valid_nl_trn = pd.read_csv(
    os.path.join(dp.valid_path, dp.valid_nl_event[0]),
    sep = ','
  )
  valid_nl_tst = pd.read_csv(
    os.path.join(dp.valid_path, dp.valid_nl_event[1]),
    sep = ','
  )
  set(valid_nl_trn.columns).difference(valid_nl_tst.columns)
  valid_nl = pd.concat([valid_nl_trn, valid_nl_tst])

  # data exception 
  if verbose:
    valid_abn.info()
    valid_abn['detection_time'].describe()
    valid_abn['detection_time'].unique()
    valid_abn['event_time'].unique()  


  detection_exception_words = ['post_op', 'post-op', 'ER/tracheostomy', \
    'x', 'ped', 'PED', 'ER', \
    'intu<24hr', 'intu<48hr', '2009-09-0913:20', 'tracheostomy',  \
    'tracheostomy', 'extubation<48hr', 'intubation state']
  valid_abn = valid_abn[~valid_abn['detection_time'].isin(detection_exception_words)]

  event_exception_words = ['ER']
  valid_abn = valid_abn[~valid_abn['event_time'].isin(event_exception_words)]
  
  valid_abn['detection_time'] = pd.to_datetime(valid_abn['detection_time'])
  valid_abn['event_time'] = valid_abn['event_time'].apply(lambda x: str(x).replace('PM', '')) 
  valid_abn['event_time'] = pd.to_datetime(valid_abn['event_time'])
  
  valid_abn['detection_time'] = valid_abn['detection_time'].dt.round('H')
  valid_abn['event_time'] = valid_abn['event_time'].dt.round('H')

  valid_event = valid_abn[['patient_id', 'gender', 'birthday', 'detection_time', 'event_time']]

  valid_nl = valid_nl[['patient_id', 'gender' , 'birthday']]  

  #save the dataset
  valid_event.to_csv(os.path.join(dp.valid_path, 'valid_event.csv'), index = False)
  # set variables

  
  return valid_abn, valid_nl, valid_event 


def main(verbose = True):

  dp = DataPath()

  valid_trn_sign = pd.read_csv(
    os.path.join(dp.valid_path, 'trn_abn_flowsheet.csv'),
    encoding = 'CP949'
  )

  valid_tst_sign = pd.read_csv(
    os.path.join(dp.valid_path, 'tst_abn_flowsheet.csv'),
    encoding = 'CP949'
  )

  valid_event = pd.read_csv(
    os.path.join(dp.valid_path, 'valid_event.csv')
  )

  valid_sign = pd.concat([valid_trn_sign, valid_tst_sign])
  valid_sign['patient_id'] = valid_sign['patient'].apply(lambda x: x[:8])

  valid_sign = valid_sign[valid_sign['patient_id'].isin(valid_event['patient_id'])]

  valid_sign = valid_sign.rename(columns={
    'Temp': 'BT'
  })  
  valid_sign = valid_sign.drop(['event_time'], axis = 1)

  valid_sign['measurement_time'] = pd.to_datetime(valid_sign['measurement_time'])
  # filter vital sign 
  valid_sign = filter_sign(valid_sign, 'patient_id', time = 'measurement_time')

  # joint two tabels
  valid_data = pd.merge(valid_sign, valid_event, left_on = 'patient_id', right_on = 'patient_id', how = 'left')
  valid_data['event_time'] = pd.to_datetime(valid_data['event_time'])
  valid_data['adjusted_time'] = pd.to_datetime(valid_data['adjusted_time'])
  valid_data['detection_time'] = pd.to_datetime(valid_data['detection_time'])
  
  # target
  valid_data = get_target_df(valid_data)
  valid_data['gender'] = valid_data['gender'].astype('category').cat.codes
  valid_data['TS'] = make_timestamp(valid_data, index = 'patient_id')
  valid_data['birthday'] = pd.to_datetime(valid_data['birthday'])
  valid_data['Age'] = (valid_data['event_time'] - valid_data['birthday']).astype('timedelta64[D]')
  valid_data['Age'] = (valid_data['Age'] / (366)).round().astype(int)
  valid_data = valid_data.drop(['patient', 'birthday'], axis = 1)
  valid_data = valid_data.rename(columns = {
    'gender': 'Gender'
  })
  print('-' * 50)
  print('CBC data')
  valid_blood_trn_cbc = pd.read_csv(
    os.path.join(dp.valid_path, 'trn_abn_cbc.csv'),
    encoding = 'CP949'
  )
  valid_blood_tst_cbc = pd.read_csv(
    os.path.join(dp.valid_path, 'tst_abn_cbc.csv'),
    encoding = 'CP949'
  )

  valid_blood_cbc = pd.concat([valid_blood_trn_cbc, valid_blood_tst_cbc])
  valid_blood_cbc['patient_id'] = valid_blood_cbc['patient'].apply(lambda x: x[:8])
  valid_blood_cbc = valid_blood_cbc[valid_blood_cbc['patient_id'].isin(valid_event['patient_id'])]
  valid_blood_cbc = valid_blood_cbc.drop(['event_time', 'measurement_time', 'patient'], axis = 1)
  
  # filter cbc value
  valid_blood_cbc = adjust_cbc(valid_blood_cbc)
  valid_blood_cbc = valid_blood_cbc.rename(columns = {
    'Platelet Count': 'platelet'
  })
  ##########
  print('-' * 50)
  print('Chem data')
  valid_blood_trn_lab = pd.read_csv(
    os.path.join(dp.valid_path, 'trn_abn_chem.csv'),
    encoding = 'CP949'
  )
  valid_blood_tst_lab = pd.read_csv(
    os.path.join(dp.valid_path, 'tst_abn_chem.csv'),
    encoding = 'CP949'
  )

  valid_blood_chem = pd.concat([valid_blood_trn_lab, valid_blood_tst_lab])
  valid_blood_chem['patient_id'] = valid_blood_chem['patient'].apply(lambda x: x[:8])
  valid_blood_chem = valid_blood_chem[valid_blood_chem['patient_id'].isin(valid_event['patient_id'])]
  valid_blood_chem = valid_blood_chem.drop(['event_time', 'measurement_time', 'patient'], axis = 1)

  # adjust chem values
  valid_blood_chem = adjust_chem(valid_blood_chem)
  valid_blood_chem = valid_blood_chem.rename(columns = {
    'Total Bilirubin': 'Total bilirubin',
    'Total Protein': 'Total protein',
    'Total Calcium': 'Total calcium',
    'Alkaline Phosphatase' : 'Alkaline phosphatase',
    'Creatinine': 'Creatinin'
  })
  if verbose:
    print('Glucose freq: ')
    print( valid_blood_chem['Glucose'].value_counts())
    print('Sodium counts: ')
    print( valid_blood_chem['Sodium'].value_counts())
    print('Potassium counts: ')
    print( valid_blood_chem['Potassium'].value_counts())
    print('Chloride counts: ')
    print( valid_blood_chem['Chloride'].value_counts())

  # merge blood data

  valid_blood = pd.merge(valid_blood_cbc, valid_blood_chem, how = 'left', \
    on = ['patient_id', 'adjusted_time'])

  blood_properties = ['WBC Count', 'Platelet Count', 'Hgb','BUN', 'creatinin', 'Glucose', 
                  'Sodium', 'Potassium', 'Chloride', 'Total protein', 'Total bilirubin',
                  'Albumin', 'CRP','Total calcium', 'Lactate', 'Alkaline phosphatase',
                  'AST', 'ALT']

  #for p in blood_properties:
  #  valid_blood[p].fillna(round(valid_blood[p].mean(), 1), inplace=True)

  valid_blood['adjusted_time'] = pd.to_datetime(valid_blood['adjusted_time'])
  valid_blood['is_abn'] = 1
  # merge blood data into valid_data
  valid_data = get_merge_data(valid_data, valid_blood)
  valid_data = valid_data.rename(columns = {
    'patient_id': 'Patient'
  })

  valid_data.to_csv(os.path.join(dp.valid_path, 'valid_merge.csv'), index = False)

  window_len = 8

  var_list = list(set(valid_data.columns)
                    - set([ 'Patient', 'adjusted_time', 'detection_time', 'measurement_time', 'event_time', 'target', 'is_abn']) )

  valid_seq_data = make_sequence_data(valid_data, window_len = window_len, var_list = var_list, index = 'Patient')
  #valid_seq_data = valid_seq_data.drop(['Patient', 'is_abn'], axis = 1)


  path = os.path.join(dp.valid_path, 'valid_seq.pickle')
  with open(path, 'wb') as f:
    pickle.dump(valid_seq_data, f)
  # Make 2D data 
  res_data = make_2d_data(valid_seq_data, var_list, \
    output_path = dp.valid_path, output_file = 'valid_final.csv')
  
  return res_data

if __name__ == '__main__':
  get_event_id()
  main()