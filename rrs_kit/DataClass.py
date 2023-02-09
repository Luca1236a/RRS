#-*-coding:utf-8
import unittest 
from dataclasses import dataclass, field 
from typing import List 

@dataclass
class DataPath:
  data_path: str = '/data/datasets/rrs-data/3yrs_raw_data'
  valid_path: str = '/data/datasets/rrs-data/10yrs_refined_data'
  output_path: str = '/data/datasets/rrs-data/3yrs_refined_data'
  model_path: str = '/data/datasets/rrs-data/trained_model'
  
  train_abn_sign_data: List[str] = field(default_factory=lambda: ['화순abn_ward.csv'])
  train_nl_sign_data: List[str] = field(default_factory=lambda: ['화순_nl_ward_외부.xlsx'])
  train_event_data: List[str] = field(default_factory=lambda: ['화순abn_list_외부.xlsx'])
  train_abn_sign_icu_data: List[str] = field(default_factory=lambda: ['cnuhh_abn_ICU.xlsx']) 

  test_abn_sign_data: List[str] = field(default_factory=lambda: \
    ['학동abn_ICU(외부).xlsx', '학동abn_list_외부.xlsx', '학동abn_ward(외부).xlsx'])
  test_nl_sign_data: List[str] = field(default_factory=lambda: ['학동nl_ward_ICU(외부).xlsx'])
  test_event_data: str = '학동abn_list_외부.xlsx'
  
  valid_abn_data: str = 'trn_abn_obj'
  valid_abn_event: str = field(default_factory = lambda: 
  ['10yrs_trn_abn.xlsx', '10yrs_tst_abn.csv'] )
  valid_nl_event: str = field( default_factory = lambda: \
    ['10yrs_trn_nl_sample.csv', '10yrs_tst_nl_sample.csv'])

  blood_data: List[str] = field(default_factory=lambda: ['외부 PD(고보건)-2019.11.7.xlsx'])
  blood_train_abn_sheet: str = 'abn(결과,화순)'
  blood_train_nl_sheet: str = 'nl(결과,화순)' 
  blood_test_abn_sheet: str = 'abn(결과,본원)'
  blood_test_nl_sheet: str = 'nl(결과,본원)'
  
  exclude_id: List[str] = field(default_factory=lambda: ['화순_학동 nl  중 제외 명단.xlsx'])
  exclude_train_sheet: str = '화순nl제외명단'
  exclude_test_sheet: str = '학동nl제외명단'

  xgb_model: List[str] = field(default_factory = lambda: \
    ['xgb_full.pickle'])
  rf_model: List[str] = field(default_factory = lambda: \
    ['rf_full.pickle']) 
  lgb_model: List[str] = field(default_factory = lambda: \
    ['lgb_full.pickle'])
  dews_model: List[str] = field(default_factory = lambda: \
    ['dews_full.pickle'])
    

@dataclass 
class VarSet:
  # Feature Sets
  stat_grp: List[str] = field(default_factory=lambda: ['mean', 'std', 'max', 'min'])
  
  time_grp: List[str] = field(default_factory=lambda: ['timestamp'])
  meta_grp: List[str] = field(default_factory=lambda: ['age', 'gender'])
  vital_grp: List[str] = field(default_factory=lambda: ['SBP', 'BT', 'SaO2', 'RR', 'HR'])
  vital_grp_stat: List[str] = field(default_factory=list)

  # CRP 항목은 없으면 삭제 
  lab_grp_a: List[str] = field(default_factory=lambda: ['WBC Count', 'Hgb', 'platelet', 'CRP'])
  lab_grp_a_stat: List[str] = field(default_factory=list)

  # 영양
  # total protein and Albumin 은 na시 초기 값으로 대체 
  # missing이 너무 많으면 제외 
  lab_grp_b: List[str] = field(default_factory=lambda: ['Total protein', 'Albumin'])
  lab_grp_b_stat: List[str] = field(default_factory=list)
 
  # 신장 및 전해질 상태
  lab_grp_c: List[str] = field(default_factory=lambda: \
    ['BUN', 'Creatinin' 'Sodium', 'Potassium', 'Chloride'])
  lab_grp_c_stat: List[str] = field(default_factory=list)

  # 간 
  # total bilirubin 은 na시 평균 대체 
  lab_grp_d: List[str] = field(default_factory=lambda: ['AST', 'ALT', 'Total bilirubin'])
  lab_grp_d_stat: List[str] = field(default_factory=list)

  lab_grp_e: List[str] = field(default_factory=lambda: \
    ['CRP', 'Creatinin'])
  #LAB_GRO_E = ['Glucose', 'Total calcium', 'Lactate', 'Alkaline phosphatase']
  lab_grp_e_stat: List[str]  = field(default_factory=list)

  lab_grp_f: List[str] = field(default_factory=lambda: \
    ['CRP'])
  lab_grp_f_stat: List[str]  = field(default_factory=list)


  lab_grp: List[str] = field(default_factory=list)
  lab_grp_stat: List[str] = field(default_factory=list)

  def __post_init__(self):
    if not self.vital_grp_stat: 
      self.vital_grp_stat = self.create_vital_grp_stat()
    
    if not self.lab_grp_a_stat:
      self.lab_grp_a_stat = self.create_lab_grp_a_stat()
    
    if not self.lab_grp_b_stat:
      self.lab_grp_b_stat = self.create_lab_grp_b_stat()
    
    if not self.lab_grp_c_stat:
      self.lab_grp_c_stat = self.create_lab_grp_c_stat()
    
    if not self.lab_grp_d_stat:
      self.lab_grp_d_stat = self.create_lab_grp_d_stat()
    
    if not self.lab_grp_e_stat:
      self.lab_grp_e_stat = self.create_lab_grp_e_stat()

    if not self.lab_grp_f_stat:
      self.lab_grp_f_stat = self.create_lab_grp_f_stat()

    if not self.lab_grp:
      self.lab_grp = self.create_lab_grp() 

    if not self.lab_grp_stat:
      self.lab_grp_stat = self.create_lab_grp_stat()

  def create_vital_grp_stat(self) -> List[str]:
    return [x + '_' + y for x in self.stat_grp for y in self.vital_grp]

  def create_lab_grp_a_stat(self) -> List[str]:
    return [x + '_' + y for x in self.stat_grp for y in self.lab_grp_a]

  def create_lab_grp_b_stat(self):
    return [x + '_' + y for x in self.stat_grp for y in self.lab_grp_b]

  def create_lab_grp_c_stat(self):
    return [x + '_' + y for x in self.stat_grp for y in self.lab_grp_c]

  def create_lab_grp_d_stat(self):
    return [x + '_' + y for x in self.stat_grp for y in self.lab_grp_d]

  def create_lab_grp_e_stat(self):
    return [x + '_' + y for x in self.stat_grp for y in self.lab_grp_e]

  def create_lab_grp_f_stat(self):
    return [x + '_' + y for x in self.stat_grp for y in self.lab_grp_f]

  def create_lab_grp(self) -> List[str]:
    return self.lab_grp_a + self.lab_grp_b + self.lab_grp_c + self.lab_grp_d 
  
  def create_lab_grp_stat(self) -> List[str]:    
    return self.lab_grp_a_stat + self.lab_grp_b_stat + self.lab_grp_c_stat + \
      self.lab_grp_d_stat 

@dataclass 
class ModelSet(VarSet):
  full: List[str] = field(default_factory=list)
  full_time: List[str] = field(default_factory=list)

  vital: List[str] = field(default_factory=list) 
  vital_time: List[str] = field(default_factory=list)

  lab_A: List[str]  = field(default_factory=list)
  lab_A_time: List[str] = field(default_factory=list)

  lab_B: List[str] = field(default_factory=list)
  lab_B_time: List[str] = field(default_factory=list)

  lab_C: List[str] = field(default_factory=list)
  lab_C_time: List[str] = field(default_factory=list)

  lab_D: List[str] = field(default_factory=list)
  lab_D_time: List[str]  = field(default_factory=list)

  lab_E: List[str] = field(default_factory=list)
  lab_E_time: List[str]  = field(default_factory=list)

  vital_lab_A: List[str] = field(default_factory=list)
  vital_lab_A_time: List[str] = field(default_factory=list)

  vital_lab_B: List[str] = field(default_factory=list)
  vital_lab_B_time: List[str] = field(default_factory=list)

  vital_lab_C: List[str] = field(default_factory=list)
  vital_lab_C_time: List[str] = field(default_factory=list)

  vital_lab_D: List[str] = field(default_factory=list)
  vital_lab_D_time: List[str] = field(default_factory=list)

  vital_lab_E: List[str] = field(default_factory=list)
  vital_lab_E_time: List[str] = field(default_factory=list)

  vital_lab_AB: List[str] = field(default_factory=list)
  vital_lab_AB_time: List[str] = field(default_factory=list)

  vital_lab_ABC: List[str] = field(default_factory=list)
  vital_lab_ABC_time: List[str] = field(default_factory=list)

  vital_lab_ABCD: List[str] = field(default_factory=list)
  vital_lab_ABCD_time: List[str] = field(default_factory=list)

  def __post_init__(self):
    if not self.full:
      self.full = self.create_full_model()
    
    if not self.full_time:
      self.full_time = self.create_full_time_model()

    if not self.vital:
      self.vital = self.create_vital_model()

    if not self.vital_time:
      self.vital_time = self.create_vital_time_model()

    if not self.lab_A:
      self.lab_A = self.create_lab_a_model() 

    if not self.lab_A_time: 
      self.lab_A_time = self.create_lab_a_time_model() 

    if not self.lab_B:
      self.lab_B = self.create_lab_b_model() 

    if not self.lab_B_time:
      self.lab_B_time = self.create_lab_b_time_model()

    if not self.lab_C:
      self.lab_C = self.create_lab_c_model() 

    if not self.lab_C_time:
      self.lab_C_time = self.create_lab_c_time_model()

    if not self.lab_D:
      self.lab_D = self.create_lab_d_model()

    if not self.lab_D_time:
      self.lab_D_time = self.create_lab_d_time_model() 

    if not self.lab_E:
      self.lab_E = self.create_lab_e_model() 

    if not self.lab_E_time: 
      self.lab_E_time = self.create_lab_e_time_model()

    if not self.vital_lab_A:
      self.vital_lab_A = self.create_vital_lab_a_model()

    if not self.vital_lab_A_time:
      self.vital_lab_A_time = self.create_vital_lab_a_time_model()

    if not self.vital_lab_B:
      self.vital_lab_B = self.create_vital_lab_b_model()

    if not self.vital_lab_B_time:
      self.vital_lab_B_time = self.create_vital_lab_b_time_model()

    if not self.vital_lab_C:
      self.vital_lab_C = self.create_vital_lab_c_model()

    if not self.vital_lab_C_time:
      self.vital_lab_C_time = self.create_vital_lab_c_time_model()

    if not self.vital_lab_D:
      self.vital_lab_D = self.create_vital_lab_d_model()

    if not self.vital_lab_D_time:
      self.vital_lab_D_time = self.create_vital_lab_d_time_model()

    if not self.vital_lab_E:
      self.vital_lab_E = self.create_vital_lab_e_model()

    if not self.vital_lab_E_time:
      self.vital_lab_E_time = self.create_vital_lab_e_time_model()

    if not self.vital_lab_AB:
      self.vital_lab_AB = self.create_vital_lab_ab_model()

    if not self.vital_lab_AB_time:
      self.vital_lab_AB_time = self.create_vital_lab_ab_time_model()

    if not self.vital_lab_AB:
      self.vital_lab_AB = self.create_vital_lab_ab_model()

    if not self.vital_lab_ABC_time:
      self.vital_lab_ABC_time = self.create_vital_lab_abc_time_model()

    if not self.vital_lab_ABCD:
      self.vital_lab_ABCD = self.create_vital_lab_abcd_model()

    if not self.vital_lab_ABCD_time:
      self.vital_lab_ABCD_time = self.create_vital_lab_abcd_time_model()

  def create_full_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.time_grp + VarSet.vital_grp + VarSet.vital_grp_stat + \
      VarSet.lab_grp_a + VarSet.lab_grp_a_stat + \
      VarSet.lab_grp_b + VarSet.lab_grp_b_stat + \
      VarSet.lab_grp_c + VarSet.lab_grp_c_stat + \
      VarSet.lab_grp_d + VarSet.lab_grp_d_stat + \
      VarSet.lab_grp_e + VarSet.lab_grp_e_stat 

  def create_full_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.time_grp + VarSet.vital_grp + VarSet.lab_grp 

  def create_vital_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp  

  def create_vital_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.time_grp

  def create_lab_a_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_a + VarSet.lab_grp_a_stat 
  
  def create_lab_a_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_a + VarSet.time_grp 

  def create_lab_b_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_b + VarSet.lab_grp_b_stat 
  
  def create_lab_b_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_b + VarSet.time_grp 

  def create_lab_c_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_c + VarSet.lab_grp_c_stat 
  
  def create_lab_c_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_c + VarSet.time_grp 

  def create_lab_d_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_d + VarSet.lab_grp_d_stat 

  def create_lab_d_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_d + VarSet.time_grp 

  def create_lab_e_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_e + VarSet.lab_grp_e_stat

  def create_lab_e_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.lab_grp_e + VarSet.time_grp 

  def create_vital_lab_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.lab_grp + VarSet.lab_grp_stat 

  def create_vital_lab_a_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.lab_grp_a + VarSet.lab_grp_a_stat

  def create_vital_lab_a_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.lab_grp_a + VarSet.time_grp 

  def create_vital_lab_b_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.lab_grp_b + VarSet.lab_grp_b_stat

  def create_vital_lab_b_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.lab_grp_b + VarSet.time_grp 

  def create_vital_lab_c_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.lab_grp_c + VarSet.lab_grp_c_stat

  def create_vital_lab_c_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.lab_grp_c + VarSet.time_grp 

  def create_vital_lab_d_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.lab_grp_d + VarSet.lab_grp_d_stat

  def create_vital_lab_d_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.lab_grp_d + VarSet.time_grp 

  def create_vital_lab_e_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + VarSet.lab_grp_e + VarSet.lab_grp_e_stat

  def create_vital_lab_e_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.lab_grp_e + VarSet.time_grp 

  def create_vital_lab_ab_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + \
      VarSet.lab_grp_a + VarSet.lab_grp_a_stat + \
      VarSet.lab_grp_b + VarSet.lab_grp_b_stat 

  def create_vital_lab_ab_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + \
      VarSet.lab_grp_a + VarSet.lab_grp_b + VarSet.time_grp 

  def create_vital_lab_abc_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + \
      VarSet.lab_grp_a + VarSet.lab_grp_a_stat + \
      VarSet.lab_grp_b + VarSet.lab_grp_b_stat + \
      VarSet.lab_grp_c + VarSet.lab_grp_c_stat 

  def create_vital_lab_abc_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + \
      VarSet.lab_grp_a + VarSet.lab_grp_b + \
      VarSet.lab_grp_c + VarSet.time_grp 

  def create_vital_lab_abcd_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.vital_grp_stat + \
      VarSet.lab_grp_a + VarSet.lab_grp_a_stat + \
      VarSet.lab_grp_b + VarSet.lab_grp_b_stat + \
      VarSet.lab_grp_c + VarSet.lab_grp_c_stat + \
      VarSet.lab_grp_d + VarSet.lab_grp_d_stat 

  def create_vital_lab_abcd_time_model(VarSet) -> List[str]:
    return VarSet.meta_grp + VarSet.vital_grp + VarSet.lab_grp_a + \
      VarSet.lab_grp_b + VarSet.time_grp + \
      VarSet.lab_grp_c + VarSet.lab_grp_d

class DataTest(unittest.TestCase):
  def test_DataPath(self) -> None:
    data = DataPath()
    print('Test DataPath')
    print(data)
    print('----')

  def test_VarSet(self) -> None:
    vars = VarSet()
    print('Test VarSet')
    print(vars)
    print('----')
  def test_ModelSet(self) -> None:
    print('Test ModelSet')
    models = ModelSet() 
    print(models)
    print('----')

if __name__=='__main__':
  unittest.main()
