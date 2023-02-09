import warnings
warnings.filterwarnings(action='ignore')
import os 
import re
import pandas as pd
import numpy as np


DATA_PATH = '/data/datasets/rrs-data/cnuhh-data'
OUTPUT_PATH = '/data/datasets/rrs-data/10yrs_refined_data'


trn_abn_obsrv = pd.read_csv(os.path.join(DATA_PATH, '시험군_TPR_임상관찰.csv'), \
  encoding = 'cp949', dtype='object')
trn_nl_obsrv = pd.read_csv(os.path.join(DATA_PATH, '대조군_TPR_임상관찰.csv', \
  encoding = 'cp949', dtype='object')
tst_abn_obsrv = pd.read_csv(os.path.join(DATA_PATH, 'TEST시험군_TPR_임상관찰.csv'), \
  encoding = 'cp949', dtype='object')
tst_nl_obsrv = pd.read_csv(os.path.join(DATA_PATH, 'TEST대조군_TPR_임상관찰.csv'), \
  encoding = 'cp949', dtype='object')

dataset_dict = {'trn_abn' : trn_abn_obsrv, 'trn_nl' : trn_nl_obsrv,
                'tst_abn' : tst_abn_obsrv, 'tst_nl' : tst_nl_obsrv}