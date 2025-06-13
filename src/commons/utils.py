import glob
import os
from pathlib import Path

from Declare4Py.ProcessModels.DeclareModel import DeclareModel

from src.ProbDeclmonitor.probDeclPredictor import ProbDeclarePredictor
from src.commons import shared_variables as shared
from src.commons.log_utils import LogData


def extract_last_model_checkpoint(log_name: str, models_folder: str, fold: int, model_type: str) -> Path:
    model_filepath = shared.output_folder / models_folder / str(fold) / 'models' / model_type / log_name
    print(f"Model filepath: {model_filepath}")  # Debugging statement

    list_of_files = glob.glob(str(model_filepath / '*.keras'))
    if not list_of_files: # add check 
        raise FileNotFoundError(f"No checkpoint files found in {model_filepath}")

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Latest checkpoint file: {latest_file}")  # Debugging statement
    return Path(latest_file)


def extract_bk_filename(log_name: str,  prefix:int) -> Path:
    if log_name.startswith('Synthetic'):
        name = shared.pn_folder / (log_name + '.bpmn')
    else:
        name = shared.pn_folder / (log_name + '_' + str(prefix) + '.bpmn')   # JH: .pnml   bpmn
    print("BK file name is:", name)
    return name

def extract_Declare_bk_model(log_name: str) -> Path:
    if log_name.startswith('Synthetic'):
        name = shared.declare_folder / ('BK_' + log_name + '.decl')
    else:
        name = shared.declare_folder / ('BK_' + log_name + str(shared.prefix_start) + '.decl')
    name = shared.declare_folder / ('BK_' + log_name + '.decl')
    print("Declare BK file name is:", name)
    declare_model = DeclareModel().parse_from_file(name)
    return declare_model

def extract_Prob_Declare_bk_model(log_name: str)-> ProbDeclarePredictor:
    modelPath = shared.declare_folder / ('BK_' + log_name.replace('_test','') + '.txt')
    print("Declare BK file name is:", modelPath)
    probDeclarePredictor = ProbDeclarePredictor()
    probDeclarePredictor.loadProbDeclModel(str(modelPath))
    return probDeclarePredictor


def extract_trace_sequences(log_data: LogData, trace_ids: list, resource: bool, outcome: bool) -> list:
    """
    Extract activity, resource and output sequences starting from a list of trace ids (i.e. trace_names).
    """
    act_seqs = []  # list of all the activity sequences
    res_seqs = []  # list of all the resource sequences
    outcomes = []  # outcome of each sequence (i.e. each case)
    
    traces = log_data.log[log_data.log[log_data.case_name_key].isin(trace_ids)]
    for _, trace in traces.groupby(log_data.case_name_key):
        line = ''.join(map(str,trace[log_data.act_name_key].tolist()))  # sequence of activities for one case
        act_seqs.append(line)

        if resource:
            line_group = ''.join(map(str,trace[log_data.res_name_key].tolist()))  # sequence of groups for one case
            res_seqs.append(line_group)
        if outcome:
            o = trace[log_data.label_name_key].iloc[0]
            outcomes.append(o)

    return act_seqs, res_seqs,  outcomes
