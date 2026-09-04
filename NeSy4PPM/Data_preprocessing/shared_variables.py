"""
This file was created in order to bring common variables and functions into one file to make code more clear
"""
from pathlib import Path

BK_type = None
mona_file_name = None
aggregationMethod= None
constraint_i = 5
beam_size = 3

############### SDFA configurations #######################
hard_pruning=False
end_distance_pruning=False
#####################################

BK_end = False
root_folder = Path(__file__).resolve().parents[2] / "docs" / "source"
data_folder = root_folder / "data"
input_folder = data_folder / "input"
output_folder = data_folder / "output"

declare_folder = input_folder / "declare_models"
log_folder = input_folder / "logs"
pn_folder = input_folder / "petrinets"

epochs = 200
train_ratio = 0.8
validation_split = 0.2

method_marker = {"SAP": "x",r"$\mathrm{NN_\mathcal{BK}}$": ".","SuTraN": "^", r"BS[$\mathrm{NN}$] (Bsize=3)": "1",
                 r"BS[$\mathrm{NN}] \rightarrow \mathcal{BK}$ (Bsize=3)": "^",r"BS[$\mathrm{NN} \ofold_times \mathcal{BK}$] (Bsize=3)": "*",
                 r"BS[$\mathrm{NN}$] (Bsize=5)": ".", r"BS[$\mathrm{NN} \ofold_times \mathcal{BK}$] (Bsize=5)":"*", r"BS[$\mathrm{NN}] \rightarrow \mathcal{BK}$ (Bsize=5)":"+",
                 r"BS[$\mathrm{NN}$] (Bsize=10)": "",r"BS[$\mathrm{NN}] \rightarrow \mathcal{BK}$ (Bsize=10)": "+", r"BS[$\mathrm{NN} \ofold_times \mathcal{BK}$] (Bsize=10)": "+"  }
method_color = {"SAP": "red","SuTraN": "brown", r"$\mathrm{NN_\mathcal{BK}}$": "magenta",
                r"BS[$\mathrm{NN}$] (Bsize=3)": "green", r"BS[$\mathrm{NN}] \rightarrow \mathcal{BK}$ (Bsize=3)": "orange",r"BS[$\mathrm{NN} \ofold_times \mathcal{BK}$] (Bsize=3)": "blue",
                r"BS[$\mathrm{NN}$] (Bsize=5)": "gray", r"{BS[$\mathrm{NN}] \rightarrow \mathcal{BK}$ (Bsize=5)":"magenta", r"BS[$\mathrm{NN} \ofold_times \mathcal{BK}$] (Bsize=5)":"cyan",
                r"BS[$\mathrm{NN}$] (Bsize=10)": "purple",r"BS[$\mathrm{NN}] \rightarrow \mathcal{BK}$ (Bsize=10)": "crimson",r"BS[$\mathrm{NN} \ofold_times \mathcal{BK}$] (Bsize=10)": "mediumpurple"} #mediumpurple



