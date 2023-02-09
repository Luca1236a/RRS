import os 
import pickle
import pandas as pd 

from DataClass import DataPath 

dp = DataPath()

valid_event = pd.read_csv(
  os.path.join(dp.valid_path, 'valid_event.csv'),
  encoding = 'CP949'
)

valid_trn_sign_icu = pd.read_csv(
  os.path.join(dp.valid_path, 'trn_abn_obsrv.csv'),
  encoding = 'CP949'
)

valid_trn_sign_icu
