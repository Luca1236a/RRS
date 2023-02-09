import os 
import pandas as pd 

DATA_PATH = '/data/datasets/rrs-data/10yrs_raw_data'
OUTPUT_PATH = '/data/datasets/rrs-data/10yrs_refined_data'

TRAIN_ABN_GENDER = '시험군_대체번호_성별추가_AIgen.csv'
TRAIN_NL_GENDER = '대조군_대체번호_성별추가_AIgen.csv' 
TEST_ABN_GENDER = 'TEST시험군_대체번호_성별추가_AIgen.csv'
TEST_NL_GENDER = 'TEST대조군_대체번호_성별추가_AIgen.csv'

def main():
  # Load dataset 
  trn_abn_gender = pd.read_csv(os.path.join(DATA_PATH, TRAIN_ABN_GENDER), encoding = 'CP949')
  trn_nl_gender = pd.read_csv(os.path.join(DATA_PATH, TRAIN_NL_GENDER), encoding = 'CP949')
  tst_abn_gender = pd.read_csv(os.path.join(DATA_PATH, TEST_ABN_GENDER), encoding = 'CP949')
  tst_nl_gender = pd.read_csv(os.path.join(DATA_PATH, TEST_NL_GENDER), encoding = 'CP949')
  
  # rename the variable 
  trn_abn_gender = trn_abn_gender.rename(columns={
    '대체번호': 'patient_id',
    '성별': 'gender',
    '생년월일': 'birthday'
  })

  trn_nl_gender = trn_nl_gender.rename(columns={
    '대체번호': 'patient_id',
    '성별': 'gender',
    '생년월일': 'birthday'
  })

  tst_abn_gender = tst_abn_gender.rename(columns={
    '대체번호': 'patient_id',
    '성별': 'gender',
    '생년월일': 'birthday'
  })

  tst_nl_gender = tst_nl_gender.rename(columns={
    '대체번호': 'patient_id',
    '성별': 'gender',
    '생년월일': 'birthday'
  })

  # save dataset
  trn_abn_gender.to_csv(os.path.join(OUTPUT_PATH, 'trn_abn_gender.csv'), index = False)
  trn_nl_gender.to_csv(os.path.join(OUTPUT_PATH, 'trn_nl_gender.csv'), index = False)
  tst_abn_gender.to_csv(os.path.join(OUTPUT_PATH, 'tst_abn_gender.csv'), index = False)
  tst_nl_gender.to_csv(os.path.join(OUTPUT_PATH, 'tst_nl_gender.csv'), index = False)
  
if __name__ == '__main__':
  main()