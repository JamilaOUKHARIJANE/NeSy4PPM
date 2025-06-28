"""
This file was created in order to bring common variables and functions into one file to make code more clear
"""
from pathlib import Path
from NeSy4PPM.ProbDeclmonitor.probDeclPredictor import AggregationMethod

BK_type = None
aggregationMethod= AggregationMethod.SUM
beam_size = 3


BK_end = False
root_folder = Path.cwd().parent /'docs'/'source'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'

declare_folder = input_folder / 'declare_models'
log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

epochs = 100
train_ratio = 0.8
validation_split = 0.2


method_marker = {'SAP': 'x','SUTRAN': '^', 'BS (bSize=3)': '1','BS + BK_END (bSize=3)': '^','BS + BK (bSize=3)': '*',
                 'BS (bSize=5)': '.', 'BS + BK (bSize=5)':'*', 'BS + BK_END (bSize=5)':'+',
                 'BS (bSize=10)': '','BS + BK_END (bSize=10)': '+', 'BS + BK (bSize=10)': '+'  }
method_color = {'SAP': 'red','SUTRAN': 'brown', 'BS (bSize=3)': 'green',
                'BS + BK_END (bSize=3)': 'orange','BS + BK (bSize=3)': 'blue',
                'BS (bSize=5)': 'gray', 'BS + BK_END (bSize=5)':'magenta', 'BS + BK (bSize=5)':'cyan',
                'BS (bSize=10)': 'purple','BS + BK_END (bSize=10)': 'crimson','BS + BK (bSize=10)': 'brown'} #mediumpurple



