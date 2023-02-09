import os 
import re 
import pandas as pd 
from pprint import pprint 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

DATA_PATH = '/data/datasets/rrs-data/10yrs_refined_data'
TEST_DATA = 'DAT03.test.csv'

test_data = pd.read_csv(os.path.join(DATA_PATH, TEST_DATA), encoding = 'CP949')

test_data.shape

test_data.head()
test_data.columns.tolist()
test_data.target
