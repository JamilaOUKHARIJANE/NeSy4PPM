"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path
from src.ProbDeclmonitor.probDeclPredictor import AggregationMethod

BK_type = None
aggregationMethod= AggregationMethod.SUM
ascii_offset = 161
beam_size = 3
prefix_start =3
th_reduction_factor = 1
One_hot_encoding=False
combined_Act_res = False
useProb_reduction = False
use_modulator = False
declare_BK = False
prob_declare_BK=False
BK_end = False
root_folder = Path.cwd() #/ 'implementation_real_logs'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output_Activity'

declare_folder = input_folder / 'declare_models'
xes_log_folder =  input_folder / 'log_prefixes'
log_folder = input_folder / 'logs'
pn_folder = input_folder / 'BPMN_models' #'petrinets'#

case_name_key = 'case:concept:name'
act_name_key = 'concept:name'
res_name_key = 'org:resource'
timestamp_key = 'time:timestamp'

epochs = 100
folds = 3
train_ratio = 0.8
variant_split = 0.9
validation_split = 0.2
iteration = 0

log_list = [
 #'Synthetic.xes',
   'helpdesk',
    'Sepsis_cases',
   #  'Road_Traffic',
    #'BPIC12',
   #'BPIC13_I',
   #'BPIC13_CP','log',
    #'Production',
    #'DomesticDeclarations',
   #'InternationalDeclarations',
    #'PermitLog',
    #'PrepaidTravelCost',
    #'RequestForPayment'

]


method_marker = {'SAP': 'x','SUTRAN': '^', 'BS (bSize=3)': '1','BS + BK_END (bSize=3)': '^','BS + BK (bSize=3)': '*',
                 'BS (bSize=5)': '.', 'BS + BK (bSize=5)':'*', 'BS + BK_END (bSize=5)':'+',
                 'BS (bSize=10)': '','BS + BK_END (bSize=10)': '+', 'BS + BK (bSize=10)': '+'  }
method_color = {'SAP': 'red','SUTRAN': 'brown', 'BS (bSize=3)': 'green',
                'BS + BK_END (bSize=3)': 'orange','BS + BK (bSize=3)': 'blue',
                'BS (bSize=5)': 'gray', 'BS + BK_END (bSize=5)':'magenta', 'BS + BK (bSize=5)':'cyan',
                'BS (bSize=10)': 'purple','BS + BK_END (bSize=10)': 'crimson','BS + BK (bSize=10)': 'brown'} #mediumpurple



