from pathlib import Path
import sys

root_dir = Path(r'')
git_dir = Path(r'')

for i in [str(root_dir), str(git_dir)]:
    sys.path.insert(0, i)

class Project(object):
    def __init__(self, root, raw_data_dir):
        self.root_dir = root
        self.rdata_dir = raw_data_dir
        self.data_dir = root / 'data'
        self.meta_dir = self.data_dir / 'meta'
        self.params_dir = self.data_dir / 'params'
        self.figs_dir = self.root_dir / 'figures'
    def create(self, raw_data=[]):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for _fp in [self.meta_dir, self.params_dir, self.figs_dir]:
            _fp.mkdir(parents=True, exist_ok=True)

my_project = Project(root_dir, root_dir / "raw_data")
my_project.create()

MMDD2dek_dict = {'01-01': 1, '01-11': 2, '01-21': 3,
                 '02-01': 4, '02-11': 5, '02-21': 6,
                 '03-01': 7, '03-11': 8, '03-21': 9,
                 '04-01': 10, '04-11': 11, '04-21': 12,
                 '05-01': 13, '05-11': 14, '05-21': 15,
                 '06-01': 16, '06-11': 17, '06-21': 18,
                 '07-01': 19, '07-11': 20, '07-21': 21,
                 '08-01': 22, '08-11': 23, '08-21': 24,
                 '09-01': 25, '09-11': 26, '09-21': 27,
                 '10-01': 28, '10-11': 29, '10-21': 30,
                 '11-01': 31, '11-11': 32, '11-21': 33,
                 '12-01': 34, '12-11': 35, '12-21': 36}
target = 'Algeria'

if target == 'Algeria':
    # step_dic = {'10-01': 1, '10-11': 2, '10-21': 3, '11-01': 4, '11-11': 5, '11-21': 6, '12-01': 7, '12-11': 8,
    #             '12-21': 9, '01-01': 10, '01-11': 11, '01-21': 12, '02-01': 13, '02-11': 14, '02-21': 15, '03-01': 16,
    #             '03-11': 17, '03-21': 18, '04-01': 19, '04-11': 20, '04-21': 21, '05-01': 22, '05-11': 23, '05-21': 24
    #             }
    first_month_in__raw_data = 8 # August; this is taken to allow data augmentation (after mirroring Oct and Nov of 2001 to Sep and Aug, all raw data start in August)
    # data are thus ordered according to a local year having index = 0 at first_month_in__raw_data
    #TODO test with same data as ML
    first_month_input_local_year = 3 # 0 is Aug, 1 is September, normally is 3 October
    first_month_analysis_local_year = 4 # December, first analysis is made 1st Dec
    n_month_analysis = 8 # Last analysis 1st July
    forecast_times = range(1,9) # forecast steps
    crop_name_ind_dict = {'Barley': 0, 'Durum wheat': 1, 'Soft wheat': 2} #this is because of Franz's code in preprocess_2D_inputs

    #calendar_dek_to_use_as_step1 = 22 #(it is first of aug, as for 2D CNN)
    #month_sos = 8


##########################################
#    ENVIRONMENT SETTINGS
##########################################
# try importing any local settings if they exist
try:
    from .local_constants import *
except ImportError as e:
    pass
