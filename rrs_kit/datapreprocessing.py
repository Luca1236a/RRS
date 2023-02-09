#-*-coding:utf-8

import os
import numpy as np 
import pandas as pd 
import datetime as dt

from pandas.core.dtypes.dtypes import str_type 

from DataClass import DataPath, VarSet 
from utils import (
  drop_row,
  weird_time,
  get_datetime, 
  get_datetime_event,
  get_target_revised,
  make_timestamp,
  fill_time,
  fill_time_nl
)

def process_sign_data(verbose = True) -> pd.DataFrame:

  verbose = True
  dp = DataPath()
  vs = VarSet() 

  if not os.path.exists(dp.output_path):
    os.mkdir(dp.output_path)
  
  # Raw data 
  trn_abn_sign = pd.read_csv(
    os.path.join(dp.data_path, dp.train_abn_sign_data[0]),
    sep = ',',
    dtype={
      '측정시각': str,
      '사망일자': str, 
      '입원일': str      
    }
  )
  trn_abn_sign['사망일자'] = pd.to_datetime(trn_abn_sign['사망일자'], format = '%Y%m%d')
  trn_abn_sign['입원일'] = pd.to_datetime(trn_abn_sign['입원일'], format = '%Y%m%d')
  trn_abn_sign = trn_abn_sign.loc[(trn_abn_sign['사망일자'] - trn_abn_sign['입원일']) <= pd.Timedelta(value=7, unit = 'd')]

  trn_abn_sign = trn_abn_sign[trn_abn_sign['대체번호'].notnull()]
  trn_abn_sign.head()
  
  trn_abn_icu_sign = pd.read_excel(
    os.path.join(dp.data_path, 'cnuhh_abn_ICU.xlsx'), #dp.test_abn_sign_data[0])
    dtype={'측정시각':str, '측정일' : str}
  )
  trn_abn_icu_sign.head()

  trn_nl_sign = pd.read_excel(
    os.path.join(dp.data_path, dp.train_nl_sign_data[0]),
    #coding = 'CP949',
    dtype = {
      '측정시각': str,
      '입원일': str, 
      '퇴원일': str 
    }
  )

  trn_nl_icu_sign = pd.read_excel(
    os.path.join(dp.data_path, 'cnuhh_nl_ICU.xlsx'), #dp.test_abn_sign_data[0])
    dtype={'측정시각':str, '측정일' : str}
  )
  trn_nl_icu_sign.head()

  trn_exclude = pd.read_excel(
    os.path.join(dp.data_path, dp.exclude_id[0]),
    sheet_name = dp.exclude_train_sheet,
    usecols = 'E, F'
  )
  trn_exclude.head()

  trn_nl_sign = trn_nl_sign.set_index('원자료번호', drop=True)
  trn_nl_sign = trn_nl_sign.drop(trn_exclude[trn_exclude['종류'] == 1]['대체번호'],axis = 0)
  trn_nl_sign = trn_nl_sign.reset_index().rename(columns={"index": "원자료번호"})


  trn_nl_sign['퇴원일'] = pd.to_datetime(trn_nl_sign['퇴원일'], format = '%Y%m%d')
  trn_nl_sign['입원일'] = pd.to_datetime(trn_nl_sign['입원일'], format = '%Y%m%d')
  trn_nl_sign = trn_nl_sign.loc[(trn_nl_sign['퇴원일'] - trn_nl_sign['입원일']) <= pd.Timedelta(value=7, unit = 'd')]
  trn_nl_sign.head()

  trn_abn_event = pd.read_excel(
    os.path.join(dp.data_path, '화순abn_list_외부.xlsx'), # dp.train_event_data[0]),
    dtype= {
      '사망일자R': str
    }
  )
  trn_abn_event.head()

  tst_abn_sign = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_ward(외부).xlsx'), #dp.test_abn_sign_data[2]),
    dtype={
      '측정시각':str, 
      '측정일' : str,
      '사망일자': str, 
      '입원일': str
    }
  )
  tst_abn_sign['사망일자'] = pd.to_datetime(tst_abn_sign['사망일자'], format = '%Y%m%d')
  tst_abn_sign['입원일'] = pd.to_datetime(tst_abn_sign['입원일'], format = '%Y%m%d')
  tst_abn_sign = tst_abn_sign.loc[(tst_abn_sign['사망일자'] - tst_abn_sign['입원일']) <= pd.Timedelta(value=7, unit = 'd')]
  tst_abn_sign.head()

  tst_abn_icu_sign = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_ICU(외부).xlsx'), #dp.test_abn_sign_data[0])
    dtype={'측정시각':str, '측정일' : str}
  )
  tst_abn_icu_sign.head()

  tst_nl_sign = pd.read_excel(
    os.path.join(dp.data_path, 'hakdong_nl_ward_ICU.xlsx'),# dp.test_nl_sign_data[0]),
    dtype={
      '측정시각':str, 
      '측정일' : str,
      '사망일자': str,
      '입원일': str
    }
  )
  tst_nl_sign['퇴원일'] = pd.to_datetime(tst_nl_sign['퇴원일'], format = '%Y%m%d')
  tst_nl_sign['입원일'] = pd.to_datetime(tst_nl_sign['입원일'], format = '%Y%m%d')
  tst_nl_sign = tst_nl_sign.loc[(tst_nl_sign['퇴원일'] - tst_nl_sign['입원일']) <= pd.Timedelta(value=7, unit = 'd')]
  tst_nl_sign.head()

  tst_abn_event = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_list_외부.xlsx') #dp.test_abn_sign_data[1])
  )
  tst_abn_event.head()

  
  # Train 코드와 PK 값이 일치 하도록 칼럼명 변경
  # Train abn pk = '대체번호'
  # Train nl pk = '원자료번호'
  # Test abn pk = '원자료번호' >>> '대체번호'
  # Test nl pk = '대체번호' >>> '원자료번호'
  
  tst_abn_sign = tst_abn_sign.rename({'원자료번호':'대체번호'}, axis='columns')
  tst_abn_icu_sign = tst_abn_icu_sign.rename({'원자료번호':'대체번호'}, axis='columns')
  
  
  #tst_nl_sign = tst_nl_sign.drop('원자료번호', axis=1)
  #tst_nl_sign = tst_nl_sign.rename({'원자료번호':'대체번호'}, axis='columns')
  #tst_nl_icu_sign = tst_nl_icu_sign.rename({'원자료번호':'대체번호'}, axis='columns')
  
  tst_abn_event = tst_abn_event.rename({
    '대체번호':'일렬번호',
    'event 일': 'event_date',
    'event time': 'event_time' ,
    'detection일': 'detection_date', 
    'detection time': 'detection_time'
    }, 
  axis='columns')


  # nl 나이값 채우기
  tst_nl_sign['나이'] = tst_nl_sign['생년월일'].apply(lambda x: 2018-int(str(x)[:4]))
  tst_nl_sign = tst_nl_sign.drop(['생년월일'], axis=1)

  # drop variables
  abn_drop_list = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '원자료번호', '중복제거후번호', '혈압_이완기']
  abn_drop_list_icu = ['성별', '생년월일', '나이', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', 
                 '퇴원일', '입원일수']
  abn_drop_list_tst = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '혈압_이완기']
  nl_drop_list = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수']
  nl_drop_list_tst = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수',
                  '혈압_이완기' ]
  event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']
  event_drop_list_tst = ['성별', '사망일자', '나이', '입원기간', '입원일', 'Unnamed: 10', 'Unnamed: 11']

  #event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']
  trn_abn_sign = trn_abn_sign.drop(abn_drop_list, axis = 1)
  trn_abn_icu_sign = trn_abn_icu_sign.drop(abn_drop_list_icu, axis = 1)
  trn_nl_sign = trn_nl_sign.drop(nl_drop_list, axis = 1)
  trn_abn_event = trn_abn_event.drop(event_drop_list, axis = 1)
  
  tst_abn_sign = tst_abn_sign.drop(abn_drop_list_tst, axis = 1)
  tst_nl_sign = tst_nl_sign.drop(nl_drop_list_tst, axis = 1)  
  tst_abn_event = tst_abn_event.drop(event_drop_list_tst, axis = 1)

  tst_nl_sign = tst_nl_sign.reset_index(drop=True)

  # convert icu data from wide to long format 
  #pd.wide_to_long(trn_abn_icu_sign, ['ICUF'], i = '대체번호', j = '측정일')

  # Get patinet ID
  trn_abn_patient = trn_abn_sign['대체번호'].drop_duplicates()
  trn_nl_patient = trn_nl_sign['원자료번호'].drop_duplicates()
  tst_abn_patient = tst_abn_sign['대체번호'].drop_duplicates()
  tst_nl_patient = tst_nl_sign['원자료번호'].drop_duplicates()  

  if verbose:
    print('훈련 데이터 총 환자 수: ', len(trn_abn_patient) + len(trn_nl_patient))
    print('   - ABN 환자 수   : ', len(trn_abn_patient))
    print('   - NL 환자 수    : ', len(trn_nl_patient))
    print('테스트 데이트 총 환자 수: ', len(tst_abn_patient) + len(tst_nl_patient))
    print('   - ABN 환자 수   : ', len(tst_abn_patient))
    print('   - NL 환자 수    : ', len(tst_nl_patient))

  # Sorting   
  abn_sort_list = ['대체번호', '측정일', '측정시각']
  nl_sort_list = ['원자료번호', '측정일', '측정시각']

  trn_abn_sign = trn_abn_sign.sort_values(abn_sort_list)
  trn_nl_sign = trn_nl_sign.sort_values(nl_sort_list)

  tst_abn_sign = tst_abn_sign.sort_values(abn_sort_list)
  tst_nl_sign = tst_nl_sign.sort_values(nl_sort_list)

  trn_nl_sign = trn_nl_sign.reset_index()
  tst_abn_sign = tst_abn_sign.reset_index()
  tst_nl_sign = tst_nl_sign.reset_index()
  
  #detecting weird data & drop row
  """
  '측정시각' 중 이상값 식별하여 제거
  1. 측정시각 중 4자리 값이 아닌 경우에 대해 제거
  2. hh/mm 으로 구분했을 경우 datetime화 될 수 없는 경우에 대해 제거
  """
  trn_abn_sign = drop_row(trn_abn_sign, drop_list = weird_time(trn_abn_sign))
  trn_nl_sign = drop_row(trn_nl_sign, drop_list = weird_time(trn_nl_sign))

  tst_abn_sign = drop_row(tst_abn_sign, drop_list = weird_time(tst_abn_sign))
  tst_nl_sign = drop_row(tst_nl_sign, drop_list = weird_time(tst_nl_sign))

  # string to datetime
  trn_abn_sign['hour'] = trn_abn_sign['측정시각'].apply(lambda x: x[0:2])
  trn_abn_sign['minute'] = trn_abn_sign['측정시각'].apply(lambda x: x[2:4])

  trn_nl_sign['hour'] = trn_nl_sign['측정시각'].apply(lambda x: x[0:2])
  trn_nl_sign['minute'] = trn_nl_sign['측정시각'].apply(lambda x: x[2:4])

  tst_abn_sign['hour'] = tst_abn_sign['측정시각'].apply(lambda x: x[0:2])
  tst_abn_sign['minute'] = tst_abn_sign['측정시각'].apply(lambda x: x[2:4])

  tst_nl_sign['hour'] = tst_nl_sign['측정시각'].apply(lambda x: x[0:2])
  tst_nl_sign['minute'] = tst_nl_sign['측정시각'].apply(lambda x: x[2:4])

  if verbose:
    print(np.unique(trn_abn_sign.hour))
    print(np.unique(trn_abn_sign.minute))

    print(np.unique(trn_nl_sign.hour))
    print(np.unique(trn_nl_sign.minute))

  weird_hour_idx = []
  weird_minute_idx = []
  for i in range(len(trn_nl_sign)):
    if trn_nl_sign.hour[i] == '  ':
      print('측정시각:',trn_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird hour↑', '-'*20)
      weird_hour_idx.append(i)
    if trn_nl_sign.minute[i] == '  ' or trn_nl_sign.minute[i] == ' 0' or trn_nl_sign.minute[i] == '0':
      print('측정시각:', trn_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird minute↑', '-'*20)
      weird_minute_idx.append(i)

  drop_list = list(set(weird_hour_idx + weird_minute_idx))
  trn_nl_sign = drop_row(trn_nl_sign, drop_list = drop_list)

  weird_hour_idx = []
  weird_minute_idx = []
  for i in range(len(tst_nl_sign)):
    if tst_nl_sign.hour[i] == '  ':
      print('측정시각:',tst_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird hour↑', '-'*20)
      weird_hour_idx.append(i)
    if tst_nl_sign.minute[i] == '  ' or tst_nl_sign.minute[i] == ' 0' or \
      tst_nl_sign.minute[i] == '0' or tst_nl_sign.minute[i] == '0\\' or \
      tst_nl_sign.minute[i] == '0 ' or tst_nl_sign.minute[i] == '':
      print('측정시각:', tst_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird minute↑', '-'*20)
      weird_minute_idx.append(i)

  drop_list = list(set(weird_hour_idx + weird_minute_idx))
  tst_nl_sign = drop_row(tst_nl_sign, drop_list = drop_list)

  if verbose:
    print('trn_abn.hour list:')
    print(np.unique(trn_abn_sign.hour))
    print('trn_abn.minute list:')
    print(np.unique(trn_abn_sign.minute))
    print('-'*20)
    print('trn_nl.hour list:')
    print(np.unique(trn_nl_sign.hour))
    print('trn_nl.minute list:')
    print(np.unique(trn_nl_sign.minute))


  #datetime 변수(측정시각) 생성
  trn_abn_sign = get_datetime(trn_abn_sign, time_var = '측정일')
  trn_nl_sign = get_datetime(trn_nl_sign, time_var = '측정일')

  tst_abn_sign = get_datetime(tst_abn_sign, time_var = '측정일')
  tst_nl_sign = get_datetime(tst_nl_sign, time_var = '측정일')

  #불필요 변수 제거
  drop_list = ['측정일', '측정시각', 'hour', 'minute']
  trn_abn_sign = trn_abn_sign.drop(drop_list, axis = 1)
  trn_nl_sign  = trn_nl_sign.drop(drop_list, axis = 1)

  tst_abn_sign = tst_abn_sign.drop(drop_list, axis = 1)
  tst_nl_sign  = tst_nl_sign.drop(drop_list, axis = 1)

  """
  측정시각 반올림
  """
  # 측정시각을 등간격으로 맞추기 위해 시간 반올림
  trn_abn_sign['adjusted_time'] = trn_abn_sign.datetime.dt.round('H')
  trn_nl_sign['adjusted_time'] = trn_nl_sign.datetime.dt.round('H')

  tst_abn_sign['adjusted_time'] = tst_abn_sign.datetime.dt.round('H')
  tst_nl_sign['adjusted_time'] = tst_nl_sign.datetime.dt.round('H')

  """
  trn_event 전처리
  1. data merge를 위해 일렬번호 앞 'abn' 제거
  2. event time 및 detection time 내 이상값 제거
  3. event time 및 detection time ==> string to datetime
  """
  
  #abn 제거
  trn_abn_event['일렬번호'] = trn_abn_event['일렬번호'].apply(lambda x: x[3:])
  
  if verbose: 
    trn_abn_event.info()
    print(pd.unique(trn_abn_event['일렬번호']))
    print('event_time')
    print(pd.unique(trn_abn_event['event_time']))
    print('-'*40)
    print('detection_time')
    print(pd.unique(trn_abn_event['detection_time']))

    #이상값 제거
    drop_event_list = []
    drop_detection_list = []
    for i in range(len(trn_abn_event)):
      if np.isnan(trn_abn_event['event_time'][i]):
        if verbose:
          print('-'*20, 'weird event_time', '-'*20)
          print('event_time:', trn_abn_event['event_time'][i])
        drop_event_list.append(i)
      if str(trn_abn_event['detection_time'][i]) == 'nan' or \
        trn_abn_event['detection_time'][i] == 'x' \
          or trn_abn_event['detection_time'][i] == '?':
        if verbose: 
          print('-'*20, 'weird detection_time', '-'*20)
          print('detection_time:', trn_abn_event['detection_time'][i])
        drop_detection_list.append(i)

  trn_abn_event = trn_abn_event[trn_abn_event['event_time'].notnull()]
  trn_abn_event = trn_abn_event[pd.to_numeric(trn_abn_event['detection_time'], errors = 'coerce').notnull()]

  tst_abn_event = tst_abn_event[tst_abn_event['event_time'].notnull()]
  tst_abn_event = tst_abn_event[pd.to_numeric(tst_abn_event['detection_time'], errors = 'coerce').notnull()]

  trn_abn_event = get_datetime_event(trn_abn_event, 'event', 'event_date', 'event_time')
  trn_abn_event = get_datetime_event(trn_abn_event, 'detection', 'detection_date', 'detection_time')

  tst_abn_event = get_datetime_event(tst_abn_event, 'event', 'event_date', 'event_time')
  tst_abn_event = get_datetime_event(tst_abn_event, 'detection', 'detection_date', 'detection_time')


  """
  merge data
  trn_abn + trn_event
  """
  trn_abn_sign['일렬번호'] = trn_abn_sign['대체번호'].apply(lambda x: x[3:])
  trn_abn_merged = trn_abn_sign.merge(trn_abn_event, left_on = '일렬번호', right_on = '일렬번호')

  tst_abn_sign['일렬번호'] = tst_abn_sign['대체번호']
  tst_abn_merged = tst_abn_sign.merge(tst_abn_event, left_on = '일렬번호', right_on = '일렬번호')

  """
  Vital Sign 이상치 확인 및 결측치 filling
  ==> 적정 데이터 Range는 고보건 선생님 메일 참조
  ==> 결측치 filling 방식
  1) 이상범위 data에 대해 null값 처리
  2) 1시간 단위로 forward filling
  3) null값에 의해 filling 되지 않은 SaO2의 경우 95로 널값 대체(아마도 평균치)
  4) 그래도 null값이 존재하는 경우(환자 한 명의 특정 컬럼 전체가 null인 경우) drop
  """
  var_list = ['혈압_수축기','체온','맥박','호흡','SaO2']

  #sign_data = pd.concat([trn_abn_sign[var_list], trn_nl_sign[var_list]])

  def filter_sign(df, index, time = 'adjusted_time', freq = '1H'):
    df['체온'].loc[df['체온'] > 43] = 43
    df['체온'].loc[df['체온'] < 35] = 35

    df['맥박'].loc[df['맥박'] > 300] = 300
    df['맥박'].loc[df['맥박'] < 30] = 30

    df['SaO2'].loc[df['SaO2'] > 100] = 100
    df['SaO2'].loc[df['SaO2'] < 65] = 65

    df['호흡'].loc[df['호흡'] > 40] = 40
    df['호흡'].loc[df['호흡'] < 6] = 6

    df['혈압_수축기'].loc[df['혈압_수축기'] > 210] = 210
    df['혈압_수축기'].loc[df['혈압_수축기'] < 60] = 60

    df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill()))
    df = df.drop([index], axis=1)
    df['SaO2'] = df['SaO2'].fillna(100)
    df = df.dropna()
    df = df.reset_index(level=[1]).reset_index()
    return df

  trn_abn_merged = filter_sign(trn_abn_merged, '일렬번호')
  trn_nl = filter_sign(trn_nl_sign, '원자료번호')

  tst_abn_merged = filter_sign(tst_abn_merged, '일렬번호')
  tst_nl = filter_sign(tst_nl_sign, '원자료번호')

  """
  target labeling
  1. abn일 경우
  - event time보다 측정시간이 나중일 경우 데이터 제거
  - time이 detection 타임과 event 타임 사이일 경우 taget = 1
  2. nl일 경우
  - 모두 0
  """

  def get_target_df(df, is_abn = True):
    if is_abn:
      df['target'] = df['event'] - df['adjusted_time']
      df = df.drop(df[df['target'] <= pd.to_timedelta(0)].index)
      df['target'] = df.apply(lambda x: get_target_revised(x['adjusted_time'], x['detection'],
                                                      x['event']), axis=1)
      return df
    else:
      df['target'] = 0
      return df 

  trn_abn = get_target_df(trn_abn_merged, is_abn=True)
  trn_nl = get_target_df(trn_nl, is_abn=False)

  tst_abn = get_target_df(tst_abn_merged, is_abn=True)
  tst_nl = get_target_df(tst_nl, is_abn=False)

  trn_abn['성별'] = trn_abn['성별'].astype('category').cat.codes #여: 1, 남: 0
  trn_nl['성별'] = trn_nl['성별'].astype('category').cat.codes

  trn_nl = trn_nl.drop(['혈압_이완기', '생년월일'], axis = 1)

  tst_abn['성별'] = tst_abn['성별'].astype('category').cat.codes #여: 1, 남: 0
  tst_nl['성별'] = tst_nl['성별'].astype('category').cat.codes

  #tst_nl = tst_nl.drop(['혈압_이완기', '생년월일'], axis = 1)

  """
  timestamp 생성
  """

  trn_abn['TS'] = make_timestamp(trn_abn)
  trn_nl['TS'] = make_timestamp(trn_nl, index = '원자료번호')

  trn_abn['type'] = 'abn'
  trn_nl['type'] = 'nl'

  trn_abn = trn_abn.rename(columns = {'일렬번호' : 'Patient'})
  trn_nl = trn_nl.rename(columns = {'원자료번호' : 'Patient'})

  tst_abn['TS'] = make_timestamp(tst_abn)
  tst_nl['TS'] = make_timestamp(tst_nl, index = '원자료번호')

  tst_abn['type'] = 'abn'
  tst_nl['type'] = 'nl'

  tst_abn = tst_abn.rename(columns = {'일렬번호' : 'Patient'})
  tst_nl = tst_nl.rename(columns = {'원자료번호' : 'Patient'})

  if verbose: 
    trn_abn.to_csv(os.path.join(dp.output_path, 'trn_abn.csv'), index=False)
    trn_nl.to_csv(os.path.join(dp.output_path, 'trn_nl.csv'), index=False)

    tst_abn.to_csv(os.path.join(dp.output_path, 'tst_abn.csv'), index = False)
    tst_nl.to_csv(os.path.join(dp.output_path, 'tst_nl.csv'), index = False)

  return trn_abn, trn_nl, tst_abn, tst_nl  

def main(verbose = True) -> pd.DataFrame:

  verbose = True
  dp = DataPath()
  vs = VarSet() 

  trn_abn = pd.read_csv(
    os.path.join(dp.output_path, 'trn_abn.csv')
  )

  trn_nl = pd.read_csv(
    os.path.join(dp.output_path, 'trn_abn.csv')
  )

  tst_abn = pd.read_csv(
    os.path.join(dp.output_path, 'tst_abn.csv')
  )

  tst_nl = pd.read_csv(
    os.path.join(dp.output_path, 'tst_nl.csv')
  )

  trn_abn_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_train_abn_sheet,
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  trn_nl_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_train_nl_sheet,
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  tst_abn_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_test_abn_sheet,
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  tst_nl_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_test_nl_sheet,
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  trn_abn_blood = trn_abn_blood.rename(columns = {'대체번호' : 'Patient'})
  trn_nl_blood = trn_nl_blood.rename(columns = {'대체번호' : 'Patient'})

  tst_abn_blood = tst_abn_blood.rename(columns = {'대체번호': 'Patient'})
  tst_nl_blood = tst_nl_blood.rename(columns = {'대체번호': 'Patient'})

  if verbose:
    trn_abn_blood.info()
    trn_nl_blood.info()

    tst_abn_blood.info()
    tst_nl_blood.info()

  trn_abn_blood = trn_abn_blood[trn_abn_blood['바코드출력일시'].notnull()]
  trn_nl_blood = trn_nl_blood[trn_nl_blood['바코드출력일시'].notnull()]

  tst_abn_blood = tst_abn_blood[tst_abn_blood['바코드출력일시'].notnull()]
  tst_nl_blood = tst_nl_blood[tst_nl_blood['바코드출력일시'].notnull()]

  if verbose:  
    pd.unique(trn_abn_blood['검사시점'])


  trn_abn_blood['바코드출력일시'] = trn_abn_blood.apply(lambda x: fill_time(x['바코드출력일시'], x['검사시점'], x['입원일'], x['detect발생1일전'], x['event발생1일전']),axis=1)

  trn_nl_blood['바코드출력일시'] = trn_nl_blood.apply(lambda x: fill_time_nl(x['바코드출력일시'],
                                                                     x['검사시점'],
                                                                     x['입원일'],
                                                                     x['입원5일차'],
                                                                     x['퇴원일'])
                                                ,axis=1)

  tst_abn_blood['바코드출력일시'] = tst_abn_blood.apply(lambda x: fill_time(x['바코드출력일시'],
                                                                    x['검사시점'],
                                                                    x['입원일'],
                                                                    x['detect발생1일전'],
                                                                    x['event발생1일전'])
                                                ,axis=1)

  tst_nl_blood['바코드출력일시'] = tst_nl_blood.apply(lambda x: fill_time_nl(x['바코드출력일시'],
                                                                     x['검사시점'],
                                                                     x['입원일'],
                                                                     x['입원5일차'],
                                                                     x['퇴원일'])
                                                ,axis=1)

  trn_abn_blood['혈액검사시점'] = pd.to_datetime(trn_abn_blood['바코드출력일시'])
  trn_nl_blood['혈액검사시점'] = pd.to_datetime(trn_nl_blood['바코드출력일시'])

  tst_abn_blood['혈액검사시점'] = pd.to_datetime(tst_abn_blood['바코드출력일시'])
  tst_nl_blood['혈액검사시점'] = pd.to_datetime(tst_nl_blood['바코드출력일시'])

  trn_abn_blood_patient = trn_abn_blood['Patient'].apply(lambda x: str(x)[3:]).unique().tolist()
  trn_abn_patient = trn_abn.Patient.apply(lambda x: str(x)).unique().tolist()

  tst_abn_blood_patient = tst_abn_blood['Patient'].unique().tolist()
  tst_abn_patient = tst_abn.Patient.unique().tolist()

  tst_nl_blood_patient = tst_nl_blood['Patient'].astype(int).astype(str).unique().tolist()
  tst_nl_patient = list(np.unique(tst_nl.Patient.apply(lambda x: str(x))))


  # Replace white space to '_' in column name
  # 혈액 데이터가 모두 비어있는 경우 라인제거
  qry  = '(ALT == ALT) or' + \
       '(BUN == BUN) or' + \
       '(Glucose == Glucose) or' + \
       '(Hgb == Hgb) or' + \
       '(Creatinin == Creatinin) or' + \
       '(Sodium == Sodium) or' + \
       '(Chloride == Chloride) or' + \
       '(Albumin == Albumin) or' + \
       '(Lactate == Lactate) or' + \
       '(AST == AST) or' + \
       '(Potassium == Potassium) or' + \
       '(CRP == CRP) or' + \
       '(platelet == platelet)'

  trn_abn_blood = trn_abn_blood.query(qry)
  trn_nl_blood = trn_nl_blood.query(qry)

  tst_abn_blood = tst_abn_blood.query(qry)
  tst_nl_blood = tst_nl_blood.query(qry)

  trn_abn_blood['Patient'] = trn_abn_blood['Patient'].apply(lambda x: str(x)[3:])
  trn_abn_blood['Patient'] = trn_abn_blood['Patient'].astype(int)
  trn_abn_blood = trn_abn_blood.sort_values(['Patient', '혈액검사시점'])

  tst_abn_blood['Patient'] = tst_abn_blood['Patient'].astype(int)
  tst_abn_blood = tst_abn_blood.sort_values(['Patient', '혈액검사시점'])

  trn_nl_blood['Patient'] = trn_nl_blood['Patient'].astype(int)
  trn_nl_blood = trn_nl_blood.sort_values(['Patient', '혈액검사시점'])

  tst_nl_blood['Patient'] = tst_nl_blood['Patient'].astype(int)
  tst_nl_blood = tst_nl_blood.sort_values(['Patient', '혈액검사시점'])

  # Reset index
  trn_abn_blood = trn_abn_blood.reset_index(drop=True)
  trn_nl_blood = trn_nl_blood.reset_index(drop=True)

  tst_abn_blood = tst_abn_blood.reset_index(drop=True)
  tst_nl_blood = tst_nl_blood.reset_index(drop=True)

  # Remove korean comment in blood values.
  blood_properties = ['WBC count', 'platelet', 'Hgb','BUN', 'Creatinin', 'Glucose', 
                  'Sodium', 'Potassium', 'Chloride', 'Total protein', 'Total bilirubin',
                  'Albumin', 'CRP','Total calcium', 'Lactate', 'Alkaline phosphatase',
                  'AST', 'ALT']

  for p in blood_properties:
    trn_abn_blood[p] = trn_abn_blood[p].astype(str)
    trn_abn_blood[p] = trn_abn_blood[p].str.extract(r'(\d+\.?\d?).*')
    trn_abn_blood[p] = trn_abn_blood[p].astype(float)
    trn_nl_blood[p] = trn_nl_blood[p].astype(str)
    trn_nl_blood[p] = trn_nl_blood[p].str.extract(r'(\d+\.?\d?).*')
    trn_nl_blood[p] = trn_nl_blood[p].astype(float)

    tst_abn_blood[p] = tst_abn_blood[p].astype(str)
    tst_abn_blood[p] = tst_abn_blood[p].str.extract(r'(\d+\.?\d?).*')
    tst_abn_blood[p] = tst_abn_blood[p].astype(float)
    tst_nl_blood[p] = tst_nl_blood[p].astype(str)
    tst_nl_blood[p] = tst_nl_blood[p].str.extract(r'(\d+\.?\d?).*')
    tst_nl_blood[p] = tst_nl_blood[p].astype(float)

  # Fill average value if value is null
  for p in blood_properties:
    trn_abn_blood[p].fillna(round(trn_abn_blood[p].mean(), 1), inplace=True)
    trn_nl_blood[p].fillna(round(trn_nl_blood[p].mean(), 1), inplace=True)

    tst_abn_blood[p].fillna(round(tst_abn_blood[p].mean(), 1), inplace=True)
    tst_nl_blood[p].fillna(round(tst_nl_blood[p].mean(), 1), inplace=True)

  abn_bld_drop_list = ['생년월일', '입원일','사망일', 'event발생1일전', 'detect발생1일전', '병원구분',
                     '검사시점', '바코드출력일시', '검체접수일시', '성별', '나이']

  nl_bld_drop_list = ['사망일', '입원일', '입원5일차', '퇴원일', '병원구분',
                    '검사시점', '바코드출력일시', '검체접수일시', '성별', '나이']

  trn_abn_blood = trn_abn_blood.drop(abn_bld_drop_list, axis = 1)
  trn_nl_blood = trn_nl_blood.drop(nl_bld_drop_list, axis = 1)

  tst_abn_blood = tst_abn_blood.drop(abn_bld_drop_list, axis = 1)
  tst_nl_blood = tst_nl_blood.drop(nl_bld_drop_list, axis = 1)

  if verbose:
    trn_abn_blood.to_csv(os.path.join(dp.output_path, 'trn_abn_blood.csv'), index = False)
    trn_nl_blood.to_csv(os.path.join(dp.output_path, 'trn_nl_blood.csv'), index = False)

    tst_abn_blood.to_csv(os.path.join(dp.output_path, 'tst_abn_blood.csv'), index = False)
    tst_nl_blood.to_csv(os.path.join(dp.output_path, 'tst_nl_blood.csv'), index = False)



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

if __name__ == '__main__':
    main() 
  # Raw data 
  trn_abn_sign = pd.read_csv(
    os.path.join(dp.data_path, dp.train_abn_sign_data[0]),
    sep = ',',
    dtype={
      '측정시각': str
    }
  )

  trn_nl_sign = pd.read_excel(
    os.path.join(dp.data_path, dp.train_nl_sign_data[0]),
    coding = 'CP949',
    dtype = {
      '측정시각': str 
    }
  )

  trn_exclude = pd.read_excel(
    os.path.join(dp.data_path, dp.exclude_id[0]),
    sheet_name = dp.exclude_train_sheet,
    usecols = 'E, F'
  )

  trn_abn_event = pd.read_excel(
    os.path.join(dp.data_path, '화순abn_list_외부.xlsx') # dp.train_event_data[0]),
  )


  tst_abn_sign = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_ward(외부).xlsx'), #dp.test_abn_sign_data[2]),
    coding = 'CP949',
    dtype={'측정시각':str, '측정일' : str}
  )

  tst_abn_sign_A = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_ICU(외부).xlsx'), #dp.test_abn_sign_data[0])
    coding = 'CP949',
    dtype={'측정시각':str, '측정일' : str}
  )

  tst_nl_sign = pd.read_excel(
    os.path.join(dp.data_path, '학동nl_ward_ICU(외부).xlsx'),# dp.test_nl_sign_data[0]),
    coding = 'CP949',
    dtype={'측정시각':str, '측정일' : str}
  )

  tst_abn_event = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_list_외부.xlsx') #dp.test_abn_sign_data[1])
  )

  # Get patinet ID
  trn_abn_patient = trn_abn_sign['대체번호'].drop_duplicates()
  trn_nl_patient = trn_nl_sign['원자료번호'].drop_duplicates()
  
  # Train 코드와 PK 값이 일치 하도록 칼럼명 변경
  # Train abn pk = '대체번호'
  # Train nl pk = '원자료번호'
  # Test abn pk = '원자료번호' >>> '대체번호'
  # Test nl pk = '대체번호' >>> '원자료번호'
  tst_abn_sign = tst_abn_sign.rename({'원자료번호':'대체번호'}, axis='columns')
  #tst_nl_sign = tst_nl_sign.drop('원자료번호', axis=1)
  #tst_nl_sign = tst_nl_sign.rename({'대체번호':'원자료번호'}, axis='columns')
  tst_abn_event = tst_abn_event.rename({
    '대체번호':'일렬번호',
    'event 일': 'event_date',
    'event time': 'event_time' ,
    'detection일': 'detection_date', 
    'detection time': 'detection_time'
    }, axis='columns')

  tst_abn_patient = tst_abn_sign['대체번호'].drop_duplicates()
  tst_nl_patient = tst_nl_sign['원자료번호'].drop_duplicates()  

  # nl 나이값 채우기
  tst_nl_sign['나이'] = tst_nl_sign['생년월일'].apply(lambda x: 2018-int(str(x)[:4]))
  tst_nl_sign = tst_nl_sign.drop(['생년월일'], axis=1)

  if verbose:
    print('훈련 데이터 총 환자 수: ', len(trn_abn_patient) + len(trn_nl_patient))
    print('   - ABN 환자 수   : ', len(trn_abn_patient))
    print('   - NL 환자 수    : ', len(trn_nl_patient))

    print('----- Train ABN Vital Sign ----- ')
    trn_abn_sign.info()

    print('----- Train NL Vital Sign -----')
    trn_nl_sign.info() 

    print('----- Test ABN Vital Sign ----- ')
    trn_abn_sign.info()

    print('----- Test NL Vital Sign -----')
    trn_nl_sign.info() 


  # drop variables
  abn_drop_list = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '원자료번호', '중복제거후번호', '혈압_이완기']
  abn_drop_list_tst = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '혈압_이완기']
  nl_drop_list = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수']
  nl_drop_list_tst = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수',
                  '혈압_이완기' ]
  event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']
  event_drop_list_tst = ['성별', '사망일자', '나이', '입원기간', '입원일', 'Unnamed: 10', 'Unnamed: 11']


  #event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']

  trn_abn_sign = trn_abn_sign.drop(abn_drop_list, axis = 1)
  trn_nl_sign = trn_nl_sign.drop(nl_drop_list, axis = 1)
  trn_abn_event = trn_abn_event.drop(event_drop_list, axis = 1)

  tst_abn_sign = tst_abn_sign.drop(abn_drop_list_tst, axis = 1)
  tst_nl_sign = tst_nl_sign.drop(nl_drop_list_tst, axis = 1)  
  tst_abn_event = tst_abn_event.drop(event_drop_list_tst, axis = 1)

  tst_nl_sign = tst_nl_sign.reset_index(drop=True)

  if verbose:
    print('Train_ABN_SIGN')
    print(trn_abn_sign.head())
    print('Train_NL_SIGN')
    print(trn_nl_sign.head())
    print('Excluded ID')
    print(trn_exclude.head())
    print('test_NL_SIGN')
    print(tst_abn_sign_A.head())
    print('test_NL_SIGN')
    print(tst_abn_sign.head())
    print('test_NL_SIGN')
    print(tst_nl_sign.head())

  trn_abn_sign = trn_abn_sign[trn_abn_sign['대체번호'].notnull()]
  
  trn_nl_sign = trn_nl_sign.set_index('원자료번호', drop=True)
  trn_nl_sign = trn_nl_sign.drop(trn_exclude[trn_exclude['종류'] == 1]['대체번호'],axis = 0)
  trn_nl_sign = trn_nl_sign.reset_index().rename(columns={"index": "원자료번호"})

  if verbose: 
    trn_nl_patient = pd.unique(trn_nl_sign['원자료번호'])
    print('제거 후 환자수:', len(trn_nl_patient))

  # Sorting   
  abn_sort_list = ['대체번호', '측정일', '측정시각']
  nl_sort_list = ['원자료번호', '측정일', '측정시각']

  trn_abn_sign = trn_abn_sign.sort_values(abn_sort_list)
  trn_nl_sign = trn_nl_sign.sort_values(nl_sort_list)

  tst_abn_sign = tst_abn_sign.sort_values(abn_sort_list)
  tst_nl_sign = tst_nl_sign.sort_values(nl_sort_list)

  #detecting weird data & drop row
  """
  '측정시각' 중 이상값 식별하여 제거
  1. 측정시각 중 4자리 값이 아닌 경우에 대해 제거
  2. hh/mm 으로 구분했을 경우 datetime화 될 수 없는 경우에 대해 제거
  """
  trn_abn_sign = drop_row(trn_abn_sign, drop_list = weird_time(trn_abn_sign))
  trn_nl_sign = drop_row(trn_nl_sign, drop_list = weird_time(trn_nl_sign))

  tst_abn_sign = drop_row(tst_abn_sign, drop_list = weird_time(tst_abn_sign))
  tst_nl_sign = drop_row(tst_nl_sign, drop_list = weird_time(tst_nl_sign))

  # string to datetime
  trn_abn_sign['hour'] = trn_abn_sign['측정시각'].apply(lambda x: x[0:2])
  trn_abn_sign['minute'] = trn_abn_sign['측정시각'].apply(lambda x: x[2:4])

  trn_nl_sign['hour'] = trn_nl_sign['측정시각'].apply(lambda x: x[0:2])
  trn_nl_sign['minute'] = trn_nl_sign['측정시각'].apply(lambda x: x[2:4])

  tst_abn_sign['hour'] = tst_abn_sign['측정시각'].apply(lambda x: x[0:2])
  tst_abn_sign['minute'] = tst_abn_sign['측정시각'].apply(lambda x: x[2:4])

  tst_nl_sign['hour'] = tst_nl_sign['측정시각'].apply(lambda x: x[0:2])
  tst_nl_sign['minute'] = tst_nl_sign['측정시각'].apply(lambda x: x[2:4])

  if verbose:
    print(np.unique(trn_abn_sign.hour))
    print(np.unique(trn_abn_sign.minute))

    print(np.unique(trn_nl_sign.hour))
    print(np.unique(trn_nl_sign.minute))

  weird_hour_idx = []
  weird_minute_idx = []
  for i in range(len(trn_nl_sign)):
    if trn_nl_sign.hour[i] == '  ':
      print('측정시각:',trn_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird hour↑', '-'*20)
      weird_hour_idx.append(i)
    if trn_nl_sign.minute[i] == '  ' or trn_nl_sign.minute[i] == ' 0' or trn_nl_sign.minute[i] == '0':
      print('측정시각:', trn_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird minute↑', '-'*20)
      weird_minute_idx.append(i)

  drop_list = list(set(weird_hour_idx + weird_minute_idx))
  trn_nl_sign = drop_row(trn_nl_sign, drop_list = drop_list)

  weird_hour_idx = []
  weird_minute_idx = []
  for i in range(len(tst_nl_sign)):
    if tst_nl_sign.hour[i] == '  ':
      print('측정시각:',tst_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird hour↑', '-'*20)
      weird_hour_idx.append(i)
    if tst_nl_sign.minute[i] == '  ' or tst_nl_sign.minute[i] == ' 0' or \
      tst_nl_sign.minute[i] == '0' or tst_nl_sign.minute[i] == '0\\' or \
      tst_nl_sign.minute[i] == '0 ' or tst_nl_sign.minute[i] == '':
      print('측정시각:', tst_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird minute↑', '-'*20)
      weird_minute_idx.append(i)

  drop_list = list(set(weird_hour_idx + weird_minute_idx))
  tst_nl_sign = drop_row(tst_nl_sign, drop_list = drop_list)

  if verbose:
    print('trn_abn.hour list:')
    print(np.unique(trn_abn_sign.hour))
    print('trn_abn.minute list:')
    print(np.unique(trn_abn_sign.minute))
    print('-'*20)
    print('trn_nl.hour list:')
    print(np.unique(trn_nl_sign.hour))
    print('trn_nl.minute list:')
    print(np.unique(trn_nl_sign.minute))

  #datetime 변수(측정시각) 생성
  trn_abn_sign = get_datetime(trn_abn_sign, time_var = '측정일')
  trn_nl_sign = get_datetime(trn_nl_sign, time_var = '측정일')

  tst_abn_sign = get_datetime(tst_abn_sign, time_var = '측정일')
  tst_nl_sign = get_datetime(tst_nl_sign, time_var = '측정일')

  #불필요 변수 제거
  drop_list = ['측정일', '측정시각', 'hour', 'minute']
  trn_abn_sign = trn_abn_sign.drop(drop_list, axis = 1)
  trn_nl_sign  = trn_nl_sign.drop(drop_list, axis = 1)

  tst_abn_sign = tst_abn_sign.drop(drop_list, axis = 1)
  tst_nl_sign  = tst_nl_sign.drop(drop_list, axis = 1)

  """
  측정시각 반올림
  """
  # 측정시각을 등간격으로 맞추기 위해 시간 반올림
  trn_abn_sign['adjusted_time'] = trn_abn_sign.datetime.dt.round('H')
  trn_nl_sign['adjusted_time'] = trn_nl_sign.datetime.dt.round('H')

  tst_abn_sign['adjusted_time'] = tst_abn_sign.datetime.dt.round('H')
  tst_nl_sign['adjusted_time'] = tst_nl_sign.datetime.dt.round('H')

  """
  trn_event 전처리
  1. data merge를 위해 일렬번호 앞 'abn' 제거
  2. event time 및 detection time 내 이상값 제거
  3. event time 및 detection time ==> string to datetime
  """
  
  #abn 제거
  trn_abn_event['일렬번호'] = trn_abn_event['일렬번호'].apply(lambda x: x[3:])
  
  if verbose: 
    trn_abn_event.info()
    print(pd.unique(trn_abn_event['일렬번호']))
    print('event_time')
    print(pd.unique(trn_abn_event['event_time']))
    print('-'*40)
    print('detection_time')
    print(pd.unique(trn_abn_event['detection_time']))

    #이상값 제거
    drop_event_list = []
    drop_detection_list = []
    for i in range(len(trn_abn_event)):
      if np.isnan(trn_abn_event['event_time'][i]):
        if verbose:
          print('-'*20, 'weird event_time', '-'*20)
          print('event_time:', trn_abn_event['event_time'][i])
        drop_event_list.append(i)
      if str(trn_abn_event['detection_time'][i]) == 'nan' or \
        trn_abn_event['detection_time'][i] == 'x' \
          or trn_abn_event['detection_time'][i] == '?':
        if verbose: 
          print('-'*20, 'weird detection_time', '-'*20)
          print('detection_time:', trn_abn_event['detection_time'][i])
        drop_detection_list.append(i)

  trn_abn_event = trn_abn_event[trn_abn_event['event_time'].notnull()]
  trn_abn_event = trn_abn_event[pd.to_numeric(trn_abn_event['detection_time'], errors = 'coerce').notnull()]

  tst_abn_event = tst_abn_event[tst_abn_event['event_time'].notnull()]
  tst_abn_event = tst_abn_event[pd.to_numeric(tst_abn_event['detection_time'], errors = 'coerce').notnull()]

  trn_abn_event = get_datetime_event(trn_abn_event, 'event', 'event_date', 'event_time')
  trn_abn_event = get_datetime_event(trn_abn_event, 'detection', 'detection_date', 'detection_time')

  tst_abn_event = get_datetime_event(tst_abn_event, 'event', 'event_date', 'event_time')
  tst_abn_event = get_datetime_event(tst_abn_event, 'detection', 'detection_date', 'detection_time')

  """
  merge data
  trn_abn + trn_event
  """
  trn_abn_sign['일렬번호'] = trn_abn_sign['대체번호'].apply(lambda x: x[3:])
  trn_abn_merged = trn_abn_sign.merge(trn_abn_event, left_on = '일렬번호', right_on = '일렬번호')

  tst_abn_sign['일렬번호'] = tst_abn_sign['대체번호']
  tst_abn_merged = tst_abn_sign.merge(tst_abn_event, left_on = '일렬번호', right_on = '일렬번호')

  """
  Vital Sign 이상치 확인 및 결측치 filling
  ==> 적정 데이터 Range는 고보건 선생님 메일 참조
  ==> 결측치 filling 방식
  1) 이상범위 data에 대해 null값 처리
  2) 1시간 단위로 forward filling
  3) null값에 의해 filling 되지 않은 SaO2의 경우 95로 널값 대체(아마도 평균치)
  4) 그래도 null값이 존재하는 경우(환자 한 명의 특정 컬럼 전체가 null인 경우) drop
  """
  var_list = ['혈압_수축기','체온','맥박','호흡','SaO2']
  #sign_data = pd.concat([trn_abn_sign[var_list], trn_nl_sign[var_list]])

  def filter_sign(df, index, time = 'adjusted_time', freq = '1H'):
    df['체온'].loc[df['체온'] > 43] = 43
    df['체온'].loc[df['체온'] < 35] = 35

    df['맥박'].loc[df['맥박'] > 300] = 300
    df['맥박'].loc[df['맥박'] < 30] = 30

    df['SaO2'].loc[df['SaO2'] > 100] = 100
    df['SaO2'].loc[df['SaO2'] < 65] = 65

    df['호흡'].loc[df['호흡'] > 40] = 40
    df['호흡'].loc[df['호흡'] < 6] = 6

    df['혈압_수축기'].loc[df['혈압_수축기'] > 210] = 210
    df['혈압_수축기'].loc[df['혈압_수축기'] < 60] = 60

    df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill()))
    df = df.drop([index], axis=1)
    df['SaO2'] = df['SaO2'].fillna(100)
    df = df.dropna()
    df = df.reset_index(level=[1]).reset_index()
    return df

  trn_abn_merged = filter_sign(trn_abn_merged, '일렬번호')
  trn_nl = filter_sign(trn_nl_sign, '원자료번호')

  tst_abn_merged = filter_sign(tst_abn_merged, '일렬번호')
  tst_nl = filter_sign(tst_nl_sign, '원자료번호')

  """
  target labeling
  1. abn일 경우
  - event time보다 측정시간이 나중일 경우 데이터 제거
  - time이 detection 타임과 event 타임 사이일 경우 taget = 1
  2. nl일 경우
  - 모두 0
  """
    
  def get_target_df(df, is_abn = True):
    if is_abn:
      df['target'] = df['event'] - df['adjusted_time']
      df = df.drop(df[df['target'] <= pd.to_timedelta(0)].index)
      df['target'] = df.apply(lambda x: get_target_revised(x['adjusted_time'], x['detection'],
                                                      x['event']), axis=1)
      return df
    else:
      df['target'] = 0
      return df 

  trn_abn = get_target_df(trn_abn_merged, is_abn=True)
  trn_nl = get_target_df(trn_nl, is_abn=False)

  tst_abn = get_target_df(tst_abn_merged, is_abn=True)
  tst_nl = get_target_df(tst_nl, is_abn=False)

  trn_abn['성별'] = trn_abn['성별'].astype('category').cat.codes #여: 1, 남: 0
  trn_nl['성별'] = trn_nl['성별'].astype('category').cat.codes

  trn_nl = trn_nl.drop(['혈압_이완기', '생년월일'], axis = 1)

  tst_abn['성별'] = tst_abn['성별'].astype('category').cat.codes #여: 1, 남: 0
  tst_nl['성별'] = tst_nl['성별'].astype('category').cat.codes

  #tst_nl = tst_nl.drop(['혈압_이완기', '생년월일'], axis = 1)

  """
  timestamp 생성
  """

  trn_abn['TS'] = make_timestamp(trn_abn)
  trn_nl['TS'] = make_timestamp(trn_nl, index = '원자료번호')

  trn_abn['type'] = 'abn'
  trn_nl['type'] = 'nl'

  trn_abn = trn_abn.rename(columns = {'일렬번호' : 'Patient'})
  trn_nl = trn_nl.rename(columns = {'원자료번호' : 'Patient'})

  tst_abn['TS'] = make_timestamp(tst_abn)
  tst_nl['TS'] = make_timestamp(tst_nl, index = '원자료번호')

  tst_abn['type'] = 'abn'
  tst_nl['type'] = 'nl'

  tst_abn = tst_abn.rename(columns = {'일렬번호' : 'Patient'})
  tst_nl = tst_nl.rename(columns = {'원자료번호' : 'Patient'})


  if verbose: 
    trn_abn.to_csv(os.path.join(dp.output_path, 'trn_abn.csv'), index=False)
    trn_nl.to_csv(os.path.join(dp.output_path, 'trn_nl.csv'), index=False)

    tst_abn.to_csv(os.path.join(dp.output_path, 'tst_abn.csv'), index = False)
    tst_nl.to_csv(os.path.join(dp.output_path, 'tst_nl.csv'), index = False)


  trn_abn_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_train_abn_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  trn_nl_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_train_nl_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  tst_abn_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_test_abn_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  tst_nl_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_test_nl_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  trn_abn_blood = trn_abn_blood.rename(columns = {'대체번호' : 'Patient'})
  trn_nl_blood = trn_nl_blood.rename(columns = {'대체번호' : 'Patient'})

  tst_abn_blood = tst_abn_blood.rename(columns = {'대체번호': 'Patient'})
  tst_nl_blood = tst_nl_blood.rename(columns = {'대체번호': 'Patient'})

  if verbose:
    trn_abn_blood.info()
    trn_nl_blood.info()

    tst_abn_blood.info()
    tst_nl_blood.info()

  trn_abn_blood = trn_abn_blood[trn_abn_blood['바코드출력일시'].notnull()]
  trn_nl_blood = trn_nl_blood[trn_nl_blood['바코드출력일시'].notnull()]

  tst_abn_blood = tst_abn_blood[tst_abn_blood['바코드출력일시'].notnull()]
  tst_nl_blood = tst_nl_blood[tst_nl_blood['바코드출력일시'].notnull()]

  if verbose:  
    pd.unique(trn_abn_blood['검사시점'])


  trn_abn_blood['바코드출력일시'] = trn_abn_blood.apply(lambda x: fill_time(x['바코드출력일시'], x['검사시점'], x['입원일'], x['detect발생1일전'], x['event발생1일전']),axis=1)

  trn_nl_blood['바코드출력일시'] = trn_nl_blood.apply(lambda x: fill_time_nl(x['바코드출력일시'],
                                                                     x['검사시점'],
                                                                     x['입원일'],
                                                                     x['입원5일차'],
                                                                     x['퇴원일'])
                                                ,axis=1)

  tst_abn_blood['바코드출력일시'] = tst_abn_blood.apply(lambda x: fill_time(x['바코드출력일시'],
                                                                    x['검사시점'],
                                                                    x['입원일'],
                                                                    x['detect발생1일전'],
                                                                    x['event발생1일전'])
                                                ,axis=1)

  tst_nl_blood['바코드출력일시'] = tst_nl_blood.apply(lambda x: fill_time_nl(x['바코드출력일시'],
                                                                     x['검사시점'],
                                                                     x['입원일'],
                                                                     x['입원5일차'],
                                                                     x['퇴원일'])
                                                ,axis=1)

  trn_abn_blood['혈액검사시점'] = pd.to_datetime(trn_abn_blood['바코드출력일시'])
  trn_nl_blood['혈액검사시점'] = pd.to_datetime(trn_nl_blood['바코드출력일시'])

  tst_abn_blood['혈액검사시점'] = pd.to_datetime(tst_abn_blood['바코드출력일시'])
  tst_nl_blood['혈액검사시점'] = pd.to_datetime(tst_nl_blood['바코드출력일시'])

  trn_abn_blood_patient = trn_abn_blood['Patient'].apply(lambda x: str(x)[3:]).unique().tolist()
  trn_abn_patient = trn_abn.Patient.apply(lambda x: str(x)).unique().tolist()

  tst_abn_blood_patient = tst_abn_blood['Patient'].unique().tolist()
  tst_abn_patient = tst_abn.Patient.unique().tolist()

  tst_nl_blood_patient = tst_nl_blood['Patient'].astype(int).astype(str).unique().tolist()
  tst_nl_patient = list(np.unique(tst_nl.Patient.apply(lambda x: str(x))))

  if verbose:
    print('<abn 사용가능 환자 수>')
    print('기존 환자 수:', len(trn_abn_patient))
    print('혈액 검사 환자 수:', len(trn_abn_blood_patient))
    print('혈액 데이터 존재하는 기존 환자 수:', len(list(set(trn_abn_patient).intersection(set(trn_abn_blood_patient)))))
    print('혈액 데이터 없는 환자:', set(trn_abn_patient).difference(set(trn_abn_blood_patient)))
    print('-'*50)
    print('<nl 사용가능 환자 수>')
    print('기존 환자 수:', len(trn_nl_patient))
    print('혈액 검사 환자 수:', len(trn_nl_blood_patient))
    print('혈액 데이터 존재하는 기존 환자 수:', len(list(set(trn_nl_patient).intersection(set(trn_nl_blood_patient)))))
    print('혈액 데이터 없는 환자:', set(trn_nl_patient).difference(set(trn_nl_blood_patient)))

  # Replace white space to '_' in column name
  # 혈액 데이터가 모두 비어있는 경우 라인제거
  qry  = '(ALT == ALT) or' + \
       '(BUN == BUN) or' + \
       '(Glucose == Glucose) or' + \
       '(Hgb == Hgb) or' + \
       '(Creatinin == Creatinin) or' + \
       '(Sodium == Sodium) or' + \
       '(Chloride == Chloride) or' + \
       '(Albumin == Albumin) or' + \
       '(Lactate == Lactate) or' + \
       '(AST == AST) or' + \
       '(Potassium == Potassium) or' + \
       '(CRP == CRP) or' + \
       '(platelet == platelet)'

  trn_abn_blood = trn_abn_blood.query(qry)
  trn_nl_blood = trn_nl_blood.query(qry)

  tst_abn_blood = tst_abn_blood.query(qry)
  tst_nl_blood = tst_nl_blood.query(qry)

  trn_abn_blood['Patient'] = trn_abn_blood['Patient'].apply(lambda x: str(x)[3:])
  trn_abn_blood['Patient'] = trn_abn_blood['Patient'].astype(int)
  trn_abn_blood = trn_abn_blood.sort_values(['Patient', '혈액검사시점'])

  tst_abn_blood['Patient'] = tst_abn_blood['Patient'].astype(int)
  tst_abn_blood = tst_abn_blood.sort_values(['Patient', '혈액검사시점'])

  trn_nl_blood['Patient'] = trn_nl_blood['Patient'].astype(int)
  trn_nl_blood = trn_nl_blood.sort_values(['Patient', '혈액검사시점'])

  tst_nl_blood['Patient'] = tst_nl_blood['Patient'].astype(int)
  tst_nl_blood = tst_nl_blood.sort_values(['Patient', '혈액검사시점'])

  # Reset index
  trn_abn_blood = trn_abn_blood.reset_index(drop=True)
  trn_nl_blood = trn_nl_blood.reset_index(drop=True)

  tst_abn_blood = tst_abn_blood.reset_index(drop=True)
  tst_nl_blood = tst_nl_blood.reset_index(drop=True)

  # Remove korean comment in blood values.
  blood_properties = ['WBC count', 'platelet', 'Hgb','BUN', 'Creatinin', 'Glucose', 
                  'Sodium', 'Potassium', 'Chloride', 'Total protein', 'Total bilirubin',
                  'Albumin', 'CRP','Total calcium', 'Lactate', 'Alkaline phosphatase',
                  'AST', 'ALT']

  for p in blood_properties:
    trn_abn_blood[p] = trn_abn_blood[p].astype(str)
    trn_abn_blood[p] = trn_abn_blood[p].str.extract(r'(\d+\.?\d?).*')
    trn_abn_blood[p] = trn_abn_blood[p].astype(float)
    trn_nl_blood[p] = trn_nl_blood[p].astype(str)
    trn_nl_blood[p] = trn_nl_blood[p].str.extract(r'(\d+\.?\d?).*')
    trn_nl_blood[p] = trn_nl_blood[p].astype(float)

    tst_abn_blood[p] = tst_abn_blood[p].astype(str)
    tst_abn_blood[p] = tst_abn_blood[p].str.extract(r'(\d+\.?\d?).*')
    tst_abn_blood[p] = tst_abn_blood[p].astype(float)
    tst_nl_blood[p] = tst_nl_blood[p].astype(str)
    tst_nl_blood[p] = tst_nl_blood[p].str.extract(r'(\d+\.?\d?).*')
    tst_nl_blood[p] = tst_nl_blood[p].astype(float)

  # Fill average value if value is null
  for p in blood_properties:
    trn_abn_blood[p].fillna(round(trn_abn_blood[p].mean(), 1), inplace=True)
    trn_nl_blood[p].fillna(round(trn_nl_blood[p].mean(), 1), inplace=True)

    tst_abn_blood[p].fillna(round(tst_abn_blood[p].mean(), 1), inplace=True)
    tst_nl_blood[p].fillna(round(tst_nl_blood[p].mean(), 1), inplace=True)

  abn_bld_drop_list = ['생년월일', '입원일','사망일', 'event발생1일전', 'detect발생1일전', '병원구분',
                     '검사시점', '바코드출력일시', '검체접수일시', '성별', '나이']

  nl_bld_drop_list = ['사망일', '입원일', '입원5일차', '퇴원일', '병원구분',
                    '검사시점', '바코드출력일시', '검체접수일시', '성별', '나이']

  trn_abn_blood = trn_abn_blood.drop(abn_bld_drop_list, axis = 1)
  trn_nl_blood = trn_nl_blood.drop(nl_bld_drop_list, axis = 1)

  tst_abn_blood = tst_abn_blood.drop(abn_bld_drop_list, axis = 1)
  tst_nl_blood = tst_nl_blood.drop(nl_bld_drop_list, axis = 1)

  if verbose:
    trn_abn_blood.to_csv(os.path.join(dp.output_path, 'trn_abn_blood.csv'), index = False)
    trn_nl_blood.to_csv(os.path.join(dp.output_path, 'trn_nl_blood.csv'), index = False)

    tst_abn_blood.to_csv(os.path.join(dp.output_path, 'tst_abn_blood.csv'), index = False)
    tst_nl_blood.to_csv(os.path.join(dp.output_path, 'tst_nl_blood.csv'), index = False)



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

if __name__ == '__main__':
    main() 
  # Raw data 
  trn_abn_sign = pd.read_csv(
    os.path.join(dp.data_path, dp.train_abn_sign_data[0]),
    sep = ',',
    dtype={
      '측정시각': str
    }
  )

  trn_nl_sign = pd.read_excel(
    os.path.join(dp.data_path, dp.train_nl_sign_data[0]),
    coding = 'CP949',
    dtype = {
      '측정시각': str 
    }
  )

  trn_exclude = pd.read_excel(
    os.path.join(dp.data_path, dp.exclude_id[0]),
    sheet_name = dp.exclude_train_sheet,
    usecols = 'E, F'
  )

  trn_abn_event = pd.read_excel(
    os.path.join(dp.data_path, '화순abn_list_외부.xlsx') # dp.train_event_data[0]),
  )


  tst_abn_sign = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_ward(외부).xlsx'), #dp.test_abn_sign_data[2]),
    coding = 'CP949',
    dtype={'측정시각':str, '측정일' : str}
  )

  tst_abn_sign_A = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_ICU(외부).xlsx'), #dp.test_abn_sign_data[0])
    coding = 'CP949',
    dtype={'측정시각':str, '측정일' : str}
  )

  tst_nl_sign = pd.read_excel(
    os.path.join(dp.data_path, '학동nl_ward_ICU(외부).xlsx'),# dp.test_nl_sign_data[0]),
    coding = 'CP949',
    dtype={'측정시각':str, '측정일' : str}
  )

  tst_abn_event = pd.read_excel(
    os.path.join(dp.data_path, '학동abn_list_외부.xlsx') #dp.test_abn_sign_data[1])
  )

  # Get patinet ID
  trn_abn_patient = trn_abn_sign['대체번호'].drop_duplicates()
  trn_nl_patient = trn_nl_sign['원자료번호'].drop_duplicates()
  
  # Train 코드와 PK 값이 일치 하도록 칼럼명 변경
  # Train abn pk = '대체번호'
  # Train nl pk = '원자료번호'
  # Test abn pk = '원자료번호' >>> '대체번호'
  # Test nl pk = '대체번호' >>> '원자료번호'
  tst_abn_sign = tst_abn_sign.rename({'원자료번호':'대체번호'}, axis='columns')
  #tst_nl_sign = tst_nl_sign.drop('원자료번호', axis=1)
  #tst_nl_sign = tst_nl_sign.rename({'대체번호':'원자료번호'}, axis='columns')
  tst_abn_event = tst_abn_event.rename({
    '대체번호':'일렬번호',
    'event 일': 'event_date',
    'event time': 'event_time' ,
    'detection일': 'detection_date', 
    'detection time': 'detection_time'
    }, axis='columns')

  tst_abn_patient = tst_abn_sign['대체번호'].drop_duplicates()
  tst_nl_patient = tst_nl_sign['원자료번호'].drop_duplicates()  

  # nl 나이값 채우기
  tst_nl_sign['나이'] = tst_nl_sign['생년월일'].apply(lambda x: 2018-int(str(x)[:4]))
  tst_nl_sign = tst_nl_sign.drop(['생년월일'], axis=1)

  if verbose:
    print('훈련 데이터 총 환자 수: ', len(trn_abn_patient) + len(trn_nl_patient))
    print('   - ABN 환자 수   : ', len(trn_abn_patient))
    print('   - NL 환자 수    : ', len(trn_nl_patient))

    print('----- Train ABN Vital Sign ----- ')
    trn_abn_sign.info()

    print('----- Train NL Vital Sign -----')
    trn_nl_sign.info() 

    print('----- Test ABN Vital Sign ----- ')
    trn_abn_sign.info()

    print('----- Test NL Vital Sign -----')
    trn_nl_sign.info() 


  # drop variables
  abn_drop_list = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '원자료번호', '중복제거후번호', '혈압_이완기']
  abn_drop_list_tst = ['생년월일', '사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일',
                 '퇴원일', '입원일수', '혈압_이완기']
  nl_drop_list = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수']
  nl_drop_list_tst = ['사망일자', '입원일', '입원이후4일', '처방일', '처방일이전4일', '퇴원일', '입원일수',
                  '혈압_이완기' ]
  event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']
  event_drop_list_tst = ['성별', '사망일자', '나이', '입원기간', '입원일', 'Unnamed: 10', 'Unnamed: 11']


  #event_drop_list = ['성별', '생년월일', '사망일자R', '사망일자', '나이', '진료일자']

  trn_abn_sign = trn_abn_sign.drop(abn_drop_list, axis = 1)
  trn_nl_sign = trn_nl_sign.drop(nl_drop_list, axis = 1)
  trn_abn_event = trn_abn_event.drop(event_drop_list, axis = 1)

  tst_abn_sign = tst_abn_sign.drop(abn_drop_list_tst, axis = 1)
  tst_nl_sign = tst_nl_sign.drop(nl_drop_list_tst, axis = 1)  
  tst_abn_event = tst_abn_event.drop(event_drop_list_tst, axis = 1)

  tst_nl_sign = tst_nl_sign.reset_index(drop=True)

  if verbose:
    print('Train_ABN_SIGN')
    print(trn_abn_sign.head())
    print('Train_NL_SIGN')
    print(trn_nl_sign.head())
    print('Excluded ID')
    print(trn_exclude.head())
    print('test_NL_SIGN')
    print(tst_abn_sign_A.head())
    print('test_NL_SIGN')
    print(tst_abn_sign.head())
    print('test_NL_SIGN')
    print(tst_nl_sign.head())

  trn_abn_sign = trn_abn_sign[trn_abn_sign['대체번호'].notnull()]
  
  trn_nl_sign = trn_nl_sign.set_index('원자료번호', drop=True)
  trn_nl_sign = trn_nl_sign.drop(trn_exclude[trn_exclude['종류'] == 1]['대체번호'],axis = 0)
  trn_nl_sign = trn_nl_sign.reset_index().rename(columns={"index": "원자료번호"})

  if verbose: 
    trn_nl_patient = pd.unique(trn_nl_sign['원자료번호'])
    print('제거 후 환자수:', len(trn_nl_patient))

  # Sorting   
  abn_sort_list = ['대체번호', '측정일', '측정시각']
  nl_sort_list = ['원자료번호', '측정일', '측정시각']

  trn_abn_sign = trn_abn_sign.sort_values(abn_sort_list)
  trn_nl_sign = trn_nl_sign.sort_values(nl_sort_list)

  tst_abn_sign = tst_abn_sign.sort_values(abn_sort_list)
  tst_nl_sign = tst_nl_sign.sort_values(nl_sort_list)

  #detecting weird data & drop row
  """
  '측정시각' 중 이상값 식별하여 제거
  1. 측정시각 중 4자리 값이 아닌 경우에 대해 제거
  2. hh/mm 으로 구분했을 경우 datetime화 될 수 없는 경우에 대해 제거
  """
  trn_abn_sign = drop_row(trn_abn_sign, drop_list = weird_time(trn_abn_sign))
  trn_nl_sign = drop_row(trn_nl_sign, drop_list = weird_time(trn_nl_sign))

  tst_abn_sign = drop_row(tst_abn_sign, drop_list = weird_time(tst_abn_sign))
  tst_nl_sign = drop_row(tst_nl_sign, drop_list = weird_time(tst_nl_sign))

  # string to datetime
  trn_abn_sign['hour'] = trn_abn_sign['측정시각'].apply(lambda x: x[0:2])
  trn_abn_sign['minute'] = trn_abn_sign['측정시각'].apply(lambda x: x[2:4])

  trn_nl_sign['hour'] = trn_nl_sign['측정시각'].apply(lambda x: x[0:2])
  trn_nl_sign['minute'] = trn_nl_sign['측정시각'].apply(lambda x: x[2:4])

  tst_abn_sign['hour'] = tst_abn_sign['측정시각'].apply(lambda x: x[0:2])
  tst_abn_sign['minute'] = tst_abn_sign['측정시각'].apply(lambda x: x[2:4])

  tst_nl_sign['hour'] = tst_nl_sign['측정시각'].apply(lambda x: x[0:2])
  tst_nl_sign['minute'] = tst_nl_sign['측정시각'].apply(lambda x: x[2:4])

  if verbose:
    print(np.unique(trn_abn_sign.hour))
    print(np.unique(trn_abn_sign.minute))

    print(np.unique(trn_nl_sign.hour))
    print(np.unique(trn_nl_sign.minute))

  weird_hour_idx = []
  weird_minute_idx = []
  for i in range(len(trn_nl_sign)):
    if trn_nl_sign.hour[i] == '  ':
      print('측정시각:',trn_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird hour↑', '-'*20)
      weird_hour_idx.append(i)
    if trn_nl_sign.minute[i] == '  ' or trn_nl_sign.minute[i] == ' 0' or trn_nl_sign.minute[i] == '0':
      print('측정시각:', trn_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird minute↑', '-'*20)
      weird_minute_idx.append(i)

  drop_list = list(set(weird_hour_idx + weird_minute_idx))
  trn_nl_sign = drop_row(trn_nl_sign, drop_list = drop_list)

  weird_hour_idx = []
  weird_minute_idx = []
  for i in range(len(tst_nl_sign)):
    if tst_nl_sign.hour[i] == '  ':
      print('측정시각:',tst_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird hour↑', '-'*20)
      weird_hour_idx.append(i)
    if tst_nl_sign.minute[i] == '  ' or tst_nl_sign.minute[i] == ' 0' or \
      tst_nl_sign.minute[i] == '0' or tst_nl_sign.minute[i] == '0\\' or \
      tst_nl_sign.minute[i] == '0 ' or tst_nl_sign.minute[i] == '':
      print('측정시각:', tst_nl_sign['측정시각'].iloc[i])
      print('index no:', i)
      print('-'*20, '↑weird minute↑', '-'*20)
      weird_minute_idx.append(i)

  drop_list = list(set(weird_hour_idx + weird_minute_idx))
  tst_nl_sign = drop_row(tst_nl_sign, drop_list = drop_list)

  if verbose:
    print('trn_abn.hour list:')
    print(np.unique(trn_abn_sign.hour))
    print('trn_abn.minute list:')
    print(np.unique(trn_abn_sign.minute))
    print('-'*20)
    print('trn_nl.hour list:')
    print(np.unique(trn_nl_sign.hour))
    print('trn_nl.minute list:')
    print(np.unique(trn_nl_sign.minute))

  #datetime 변수(측정시각) 생성
  trn_abn_sign = get_datetime(trn_abn_sign, time_var = '측정일')
  trn_nl_sign = get_datetime(trn_nl_sign, time_var = '측정일')

  tst_abn_sign = get_datetime(tst_abn_sign, time_var = '측정일')
  tst_nl_sign = get_datetime(tst_nl_sign, time_var = '측정일')

  #불필요 변수 제거
  drop_list = ['측정일', '측정시각', 'hour', 'minute']
  trn_abn_sign = trn_abn_sign.drop(drop_list, axis = 1)
  trn_nl_sign  = trn_nl_sign.drop(drop_list, axis = 1)

  tst_abn_sign = tst_abn_sign.drop(drop_list, axis = 1)
  tst_nl_sign  = tst_nl_sign.drop(drop_list, axis = 1)

  """
  측정시각 반올림
  """
  # 측정시각을 등간격으로 맞추기 위해 시간 반올림
  trn_abn_sign['adjusted_time'] = trn_abn_sign.datetime.dt.round('H')
  trn_nl_sign['adjusted_time'] = trn_nl_sign.datetime.dt.round('H')

  tst_abn_sign['adjusted_time'] = tst_abn_sign.datetime.dt.round('H')
  tst_nl_sign['adjusted_time'] = tst_nl_sign.datetime.dt.round('H')

  """
  trn_event 전처리
  1. data merge를 위해 일렬번호 앞 'abn' 제거
  2. event time 및 detection time 내 이상값 제거
  3. event time 및 detection time ==> string to datetime
  """
  
  #abn 제거
  trn_abn_event['일렬번호'] = trn_abn_event['일렬번호'].apply(lambda x: x[3:])
  
  if verbose: 
    trn_abn_event.info()
    print(pd.unique(trn_abn_event['일렬번호']))
    print('event_time')
    print(pd.unique(trn_abn_event['event_time']))
    print('-'*40)
    print('detection_time')
    print(pd.unique(trn_abn_event['detection_time']))

    #이상값 제거
    drop_event_list = []
    drop_detection_list = []
    for i in range(len(trn_abn_event)):
      if np.isnan(trn_abn_event['event_time'][i]):
        if verbose:
          print('-'*20, 'weird event_time', '-'*20)
          print('event_time:', trn_abn_event['event_time'][i])
        drop_event_list.append(i)
      if str(trn_abn_event['detection_time'][i]) == 'nan' or \
        trn_abn_event['detection_time'][i] == 'x' \
          or trn_abn_event['detection_time'][i] == '?':
        if verbose: 
          print('-'*20, 'weird detection_time', '-'*20)
          print('detection_time:', trn_abn_event['detection_time'][i])
        drop_detection_list.append(i)

  trn_abn_event = trn_abn_event[trn_abn_event['event_time'].notnull()]
  trn_abn_event = trn_abn_event[pd.to_numeric(trn_abn_event['detection_time'], errors = 'coerce').notnull()]

  tst_abn_event = tst_abn_event[tst_abn_event['event_time'].notnull()]
  tst_abn_event = tst_abn_event[pd.to_numeric(tst_abn_event['detection_time'], errors = 'coerce').notnull()]

  trn_abn_event = get_datetime_event(trn_abn_event, 'event', 'event_date', 'event_time')
  trn_abn_event = get_datetime_event(trn_abn_event, 'detection', 'detection_date', 'detection_time')

  tst_abn_event = get_datetime_event(tst_abn_event, 'event', 'event_date', 'event_time')
  tst_abn_event = get_datetime_event(tst_abn_event, 'detection', 'detection_date', 'detection_time')

  """
  merge data
  trn_abn + trn_event
  """
  trn_abn_sign['일렬번호'] = trn_abn_sign['대체번호'].apply(lambda x: x[3:])
  trn_abn_merged = trn_abn_sign.merge(trn_abn_event, left_on = '일렬번호', right_on = '일렬번호')

  tst_abn_sign['일렬번호'] = tst_abn_sign['대체번호']
  tst_abn_merged = tst_abn_sign.merge(tst_abn_event, left_on = '일렬번호', right_on = '일렬번호')

  """
  Vital Sign 이상치 확인 및 결측치 filling
  ==> 적정 데이터 Range는 고보건 선생님 메일 참조
  ==> 결측치 filling 방식
  1) 이상범위 data에 대해 null값 처리
  2) 1시간 단위로 forward filling
  3) null값에 의해 filling 되지 않은 SaO2의 경우 95로 널값 대체(아마도 평균치)
  4) 그래도 null값이 존재하는 경우(환자 한 명의 특정 컬럼 전체가 null인 경우) drop
  """
  var_list = ['혈압_수축기','체온','맥박','호흡','SaO2']
  #sign_data = pd.concat([trn_abn_sign[var_list], trn_nl_sign[var_list]])

  def filter_sign(df, index, time = 'adjusted_time', freq = '1H'):
    df['체온'].loc[df['체온'] > 43] = 43
    df['체온'].loc[df['체온'] < 35] = 35

    df['맥박'].loc[df['맥박'] > 300] = 300
    df['맥박'].loc[df['맥박'] < 30] = 30

    df['SaO2'].loc[df['SaO2'] > 100] = 100
    df['SaO2'].loc[df['SaO2'] < 65] = 65

    df['호흡'].loc[df['호흡'] > 40] = 40
    df['호흡'].loc[df['호흡'] < 6] = 6

    df['혈압_수축기'].loc[df['혈압_수축기'] > 210] = 210
    df['혈압_수축기'].loc[df['혈압_수축기'] < 60] = 60

    df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill()))
    df = df.drop([index], axis=1)
    df['SaO2'] = df['SaO2'].fillna(100)
    df = df.dropna()
    df = df.reset_index(level=[1]).reset_index()
    return df

  trn_abn_merged = filter_sign(trn_abn_merged, '일렬번호')
  trn_nl = filter_sign(trn_nl_sign, '원자료번호')

  tst_abn_merged = filter_sign(tst_abn_merged, '일렬번호')
  tst_nl = filter_sign(tst_nl_sign, '원자료번호')

  """
  target labeling
  1. abn일 경우
  - event time보다 측정시간이 나중일 경우 데이터 제거
  - time이 detection 타임과 event 타임 사이일 경우 taget = 1
  2. nl일 경우
  - 모두 0
  """
    
  def get_target_df(df, is_abn = True):
    if is_abn:
      df['target'] = df['event'] - df['adjusted_time']
      df = df.drop(df[df['target'] <= pd.to_timedelta(0)].index)
      df['target'] = df.apply(lambda x: get_target_revised(x['adjusted_time'], x['detection'],
                                                      x['event']), axis=1)
      return df
    else:
      df['target'] = 0
      return df 

  trn_abn = get_target_df(trn_abn_merged, is_abn=True)
  trn_nl = get_target_df(trn_nl, is_abn=False)

  tst_abn = get_target_df(tst_abn_merged, is_abn=True)
  tst_nl = get_target_df(tst_nl, is_abn=False)

  trn_abn['성별'] = trn_abn['성별'].astype('category').cat.codes #여: 1, 남: 0
  trn_nl['성별'] = trn_nl['성별'].astype('category').cat.codes

  trn_nl = trn_nl.drop(['혈압_이완기', '생년월일'], axis = 1)

  tst_abn['성별'] = tst_abn['성별'].astype('category').cat.codes #여: 1, 남: 0
  tst_nl['성별'] = tst_nl['성별'].astype('category').cat.codes

  #tst_nl = tst_nl.drop(['혈압_이완기', '생년월일'], axis = 1)

  """
  timestamp 생성
  """

  trn_abn['TS'] = make_timestamp(trn_abn)
  trn_nl['TS'] = make_timestamp(trn_nl, index = '원자료번호')

  trn_abn['type'] = 'abn'
  trn_nl['type'] = 'nl'

  trn_abn = trn_abn.rename(columns = {'일렬번호' : 'Patient'})
  trn_nl = trn_nl.rename(columns = {'원자료번호' : 'Patient'})

  tst_abn['TS'] = make_timestamp(tst_abn)
  tst_nl['TS'] = make_timestamp(tst_nl, index = '원자료번호')

  tst_abn['type'] = 'abn'
  tst_nl['type'] = 'nl'

  tst_abn = tst_abn.rename(columns = {'일렬번호' : 'Patient'})
  tst_nl = tst_nl.rename(columns = {'원자료번호' : 'Patient'})


  if verbose: 
    trn_abn.to_csv(os.path.join(dp.output_path, 'trn_abn.csv'), index=False)
    trn_nl.to_csv(os.path.join(dp.output_path, 'trn_nl.csv'), index=False)

    tst_abn.to_csv(os.path.join(dp.output_path, 'tst_abn.csv'), index = False)
    tst_nl.to_csv(os.path.join(dp.output_path, 'tst_nl.csv'), index = False)


  trn_abn_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_train_abn_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  trn_nl_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_train_nl_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  tst_abn_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_test_abn_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  tst_nl_blood = pd.read_excel( 
    io = os.path.join(dp.data_path, dp.blood_data[0]),
    sheet_name= dp.blood_test_nl_sheet,
    coding = 'CP949',
    dtype = {'바코드출력일시' : 'str', '검체접수일시' : 'str'}
  )

  trn_abn_blood = trn_abn_blood.rename(columns = {'대체번호' : 'Patient'})
  trn_nl_blood = trn_nl_blood.rename(columns = {'대체번호' : 'Patient'})

  tst_abn_blood = tst_abn_blood.rename(columns = {'대체번호': 'Patient'})
  tst_nl_blood = tst_nl_blood.rename(columns = {'대체번호': 'Patient'})

  if verbose:
    trn_abn_blood.info()
    trn_nl_blood.info()

    tst_abn_blood.info()
    tst_nl_blood.info()

  trn_abn_blood = trn_abn_blood[trn_abn_blood['바코드출력일시'].notnull()]
  trn_nl_blood = trn_nl_blood[trn_nl_blood['바코드출력일시'].notnull()]

  tst_abn_blood = tst_abn_blood[tst_abn_blood['바코드출력일시'].notnull()]
  tst_nl_blood = tst_nl_blood[tst_nl_blood['바코드출력일시'].notnull()]

  if verbose:  
    pd.unique(trn_abn_blood['검사시점'])


  trn_abn_blood['바코드출력일시'] = trn_abn_blood.apply(lambda x: fill_time(x['바코드출력일시'], x['검사시점'], x['입원일'], x['detect발생1일전'], x['event발생1일전']),axis=1)

  trn_nl_blood['바코드출력일시'] = trn_nl_blood.apply(lambda x: fill_time_nl(x['바코드출력일시'],
                                                                     x['검사시점'],
                                                                     x['입원일'],
                                                                     x['입원5일차'],
                                                                     x['퇴원일'])
                                                ,axis=1)

  tst_abn_blood['바코드출력일시'] = tst_abn_blood.apply(lambda x: fill_time(x['바코드출력일시'],
                                                                    x['검사시점'],
                                                                    x['입원일'],
                                                                    x['detect발생1일전'],
                                                                    x['event발생1일전'])
                                                ,axis=1)

  tst_nl_blood['바코드출력일시'] = tst_nl_blood.apply(lambda x: fill_time_nl(x['바코드출력일시'],
                                                                     x['검사시점'],
                                                                     x['입원일'],
                                                                     x['입원5일차'],
                                                                     x['퇴원일'])
                                                ,axis=1)

  trn_abn_blood['혈액검사시점'] = pd.to_datetime(trn_abn_blood['바코드출력일시'])
  trn_nl_blood['혈액검사시점'] = pd.to_datetime(trn_nl_blood['바코드출력일시'])

  tst_abn_blood['혈액검사시점'] = pd.to_datetime(tst_abn_blood['바코드출력일시'])
  tst_nl_blood['혈액검사시점'] = pd.to_datetime(tst_nl_blood['바코드출력일시'])

  trn_abn_blood_patient = trn_abn_blood['Patient'].apply(lambda x: str(x)[3:]).unique().tolist()
  trn_abn_patient = trn_abn.Patient.apply(lambda x: str(x)).unique().tolist()

  tst_abn_blood_patient = tst_abn_blood['Patient'].unique().tolist()
  tst_abn_patient = tst_abn.Patient.unique().tolist()

  tst_nl_blood_patient = tst_nl_blood['Patient'].astype(int).astype(str).unique().tolist()
  tst_nl_patient = list(np.unique(tst_nl.Patient.apply(lambda x: str(x))))

  if verbose:
    print('<abn 사용가능 환자 수>')
    print('기존 환자 수:', len(trn_abn_patient))
    print('혈액 검사 환자 수:', len(trn_abn_blood_patient))
    print('혈액 데이터 존재하는 기존 환자 수:', len(list(set(trn_abn_patient).intersection(set(trn_abn_blood_patient)))))
    print('혈액 데이터 없는 환자:', set(trn_abn_patient).difference(set(trn_abn_blood_patient)))
    print('-'*50)
    print('<nl 사용가능 환자 수>')
    print('기존 환자 수:', len(trn_nl_patient))
    print('혈액 검사 환자 수:', len(trn_nl_blood_patient))
    print('혈액 데이터 존재하는 기존 환자 수:', len(list(set(trn_nl_patient).intersection(set(trn_nl_blood_patient)))))
    print('혈액 데이터 없는 환자:', set(trn_nl_patient).difference(set(trn_nl_blood_patient)))

  # Replace white space to '_' in column name
  # 혈액 데이터가 모두 비어있는 경우 라인제거
  qry  = '(ALT == ALT) or' + \
       '(BUN == BUN) or' + \
       '(Glucose == Glucose) or' + \
       '(Hgb == Hgb) or' + \
       '(Creatinin == Creatinin) or' + \
       '(Sodium == Sodium) or' + \
       '(Chloride == Chloride) or' + \
       '(Albumin == Albumin) or' + \
       '(Lactate == Lactate) or' + \
       '(AST == AST) or' + \
       '(Potassium == Potassium) or' + \
       '(CRP == CRP) or' + \
       '(platelet == platelet)'

  trn_abn_blood = trn_abn_blood.query(qry)
  trn_nl_blood = trn_nl_blood.query(qry)

  tst_abn_blood = tst_abn_blood.query(qry)
  tst_nl_blood = tst_nl_blood.query(qry)

  trn_abn_blood['Patient'] = trn_abn_blood['Patient'].apply(lambda x: str(x)[3:])
  trn_abn_blood['Patient'] = trn_abn_blood['Patient'].astype(int)
  trn_abn_blood = trn_abn_blood.sort_values(['Patient', '혈액검사시점'])

  tst_abn_blood['Patient'] = tst_abn_blood['Patient'].astype(int)
  tst_abn_blood = tst_abn_blood.sort_values(['Patient', '혈액검사시점'])

  trn_nl_blood['Patient'] = trn_nl_blood['Patient'].astype(int)
  trn_nl_blood = trn_nl_blood.sort_values(['Patient', '혈액검사시점'])

  tst_nl_blood['Patient'] = tst_nl_blood['Patient'].astype(int)
  tst_nl_blood = tst_nl_blood.sort_values(['Patient', '혈액검사시점'])

  # Reset index
  trn_abn_blood = trn_abn_blood.reset_index(drop=True)
  trn_nl_blood = trn_nl_blood.reset_index(drop=True)

  tst_abn_blood = tst_abn_blood.reset_index(drop=True)
  tst_nl_blood = tst_nl_blood.reset_index(drop=True)

  # Remove korean comment in blood values.
  blood_properties = ['WBC count', 'platelet', 'Hgb','BUN', 'Creatinin', 'Glucose', 
                  'Sodium', 'Potassium', 'Chloride', 'Total protein', 'Total bilirubin',
                  'Albumin', 'CRP','Total calcium', 'Lactate', 'Alkaline phosphatase',
                  'AST', 'ALT']

  for p in blood_properties:
    trn_abn_blood[p] = trn_abn_blood[p].astype(str)
    trn_abn_blood[p] = trn_abn_blood[p].str.extract(r'(\d+\.?\d?).*')
    trn_abn_blood[p] = trn_abn_blood[p].astype(float)
    trn_nl_blood[p] = trn_nl_blood[p].astype(str)
    trn_nl_blood[p] = trn_nl_blood[p].str.extract(r'(\d+\.?\d?).*')
    trn_nl_blood[p] = trn_nl_blood[p].astype(float)

    tst_abn_blood[p] = tst_abn_blood[p].astype(str)
    tst_abn_blood[p] = tst_abn_blood[p].str.extract(r'(\d+\.?\d?).*')
    tst_abn_blood[p] = tst_abn_blood[p].astype(float)
    tst_nl_blood[p] = tst_nl_blood[p].astype(str)
    tst_nl_blood[p] = tst_nl_blood[p].str.extract(r'(\d+\.?\d?).*')
    tst_nl_blood[p] = tst_nl_blood[p].astype(float)

  # Fill average value if value is null
  for p in blood_properties:
    trn_abn_blood[p].fillna(round(trn_abn_blood[p].mean(), 1), inplace=True)
    trn_nl_blood[p].fillna(round(trn_nl_blood[p].mean(), 1), inplace=True)

    tst_abn_blood[p].fillna(round(tst_abn_blood[p].mean(), 1), inplace=True)
    tst_nl_blood[p].fillna(round(tst_nl_blood[p].mean(), 1), inplace=True)

  abn_bld_drop_list = ['생년월일', '입원일','사망일', 'event발생1일전', 'detect발생1일전', '병원구분',
                     '검사시점', '바코드출력일시', '검체접수일시', '성별', '나이']

  nl_bld_drop_list = ['사망일', '입원일', '입원5일차', '퇴원일', '병원구분',
                    '검사시점', '바코드출력일시', '검체접수일시', '성별', '나이']

  trn_abn_blood = trn_abn_blood.drop(abn_bld_drop_list, axis = 1)
  trn_nl_blood = trn_nl_blood.drop(nl_bld_drop_list, axis = 1)

  tst_abn_blood = tst_abn_blood.drop(abn_bld_drop_list, axis = 1)
  tst_nl_blood = tst_nl_blood.drop(nl_bld_drop_list, axis = 1)

  if verbose:
    trn_abn_blood.to_csv(os.path.join(dp.output_path, 'trn_abn_blood.csv'), index = False)
    trn_nl_blood.to_csv(os.path.join(dp.output_path, 'trn_nl_blood.csv'), index = False)

    tst_abn_blood.to_csv(os.path.join(dp.output_path, 'tst_abn_blood.csv'), index = False)
    tst_nl_blood.to_csv(os.path.join(dp.output_path, 'tst_nl_blood.csv'), index = False)



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

if __name__ == '__main__':
    main() 