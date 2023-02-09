import os 
import pandas as pd 
import datetime as dt
from DataClass import DataPath, VarSet 

dp = DataPath()

trn_abn = pd.read_csv(os.path.join(dp.output_path, 'trn_abn.csv'))
trn_abn = trn_abn.rename(columns = {'일렬번호' : 'Patient'})
trn_abn['adjusted_time'] = pd.to_datetime(trn_abn['adjusted_time'])

trn_abn_bld = pd.read_csv(os.path.join(dp.output_path, 'trn_abn_blood.csv'))
trn_abn_bld['adjusted_time'] = pd.to_datetime(trn_abn_bld['혈액검사시점']).dt.round('H')

trn_nl = pd.read_csv(os.path.join(dp.output_path, 'trn_nl.csv'))
trn_nl = trn_nl.rename(columns = {'원자료번호' : 'Patient'})
trn_nl['adjusted_time'] = pd.to_datetime(trn_nl['adjusted_time'])

trn_nl_bld = pd.read_csv(os.path.join(dp.output_path, 'trn_nl_blood.csv'))
trn_nl_bld['adjusted_time'] = pd.to_datetime(trn_nl_bld['혈액검사시점']).dt.round('H')

tst_abn = pd.read_csv(os.path.join(dp.output_path, 'tst_abn.csv'))
tst_abn = tst_abn.rename(columns = {'일렬번호' : 'Patient'})
tst_abn['adjusted_time'] = pd.to_datetime(tst_abn['adjusted_time'])

tst_abn_bld = pd.read_csv(os.path.join(dp.output_path, 'tst_abn_blood.csv'))
tst_abn_bld['adjusted_time'] = pd.to_datetime(tst_abn_bld['혈액검사시점']).dt.round('H')

tst_nl = pd.read_csv(os.path.join(dp.output_path, 'tst_nl.csv'))
tst_nl = tst_nl.rename(columns = {'원자료번호' : 'Patient'})
tst_nl['adjusted_time'] = pd.to_datetime(tst_nl['adjusted_time'])

tst_nl_bld = pd.read_csv(os.path.join(dp.output_path, 'tst_nl_blood.csv'))
tst_nl_bld['adjusted_time'] = pd.to_datetime(tst_nl_bld['혈액검사시점']).dt.round('H')

def get_resampled(df, index, time = 'adjusted_time', freq = '1H', without_bfill = True):
    if without_bfill:
        df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill()))
        df = df.drop([index], axis=1)
        #df = df.dropna()
        df = df.reset_index(level=[1]).reset_index()
    else:
        df = (df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill().bfill()))
        df = df.drop([index], axis=1)
        #df = df.dropna()
        df = df.reset_index(level=[1]).reset_index()       
    return df

def get_merge_data(df, df_bld):
    df_bld = get_resampled(df_bld, 'Patient')
    df_merged = df.merge(df_bld,
                         left_on = ['Patient', 'adjusted_time'],
                         right_on = ['Patient', 'adjusted_time'],
                         how = 'left')
    df_merged = get_resampled(df_merged, 'Patient', without_bfill=False)
    return df_merged

trn_abn_merged = get_merge_data(trn_abn, trn_abn_bld)
trn_nl_merged = get_merge_data(trn_nl, trn_nl_bld)

tst_abn_merged = get_merge_data(tst_abn, tst_abn_bld)
tst_nl_merged = get_merge_data(tst_nl, tst_nl_bld)

trn_abn_merged.to_csv(os.path.join(dp.output_path, 'trn_abn_merged.csv'), index = False, encoding = 'utf-8')
trn_nl_merged.to_csv(os.path.join(dp.output_path, 'trn_nl_merged.csv'), index = False, encoding = 'utf-8')
tst_abn_merged.to_csv(os.path.join(dp.output_path, 'tst_abn_merged.csv'), index = False, encoding = 'utf-8')
tst_nl_merged.to_csv(os.path.join(dp.output_path, 'tst_nl_merged.csv'), index = False, encoding = 'utf-8')