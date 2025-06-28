import glob
import os
from enum import Enum
from pathlib import Path
import copy

import pm4py
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from NeSy4PPM.ProbDeclmonitor.probDeclPredictor import ProbDeclarePredictor
from NeSy4PPM.Data_preprocessing import shared_variables as shared
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from pm4py.visualization.petri_net import visualizer as vis_factory

class BK_type(Enum):
    Procedural = 'Procedural'
    Declare = '(MP)Declare'
    ProbDeclare = 'ProbDeclare'
    Procedural_End = 'Procedural_At_end'
    Declare_End = 'Declare_At_end'

class NN_model(Enum):
    LSTM = 'LSTM'
    Transformer = 'keras_trans'

class Encodings(Enum):
    One_hot = 'one-hot'
    Index_based = 'index-based'
    Shrinked_based = 'shrinked index-based'
    Multi_encoders = 'multi-encoders'


def extract_last_model_checkpoint(log_name: str, models_folder:str, model_type: str,ckeckpoint_folder=shared.output_folder) -> Path:
    model_filepath = ckeckpoint_folder / models_folder / 'models' / model_type / log_name
    print(f"Model filepath: {model_filepath}")  # Debugging statement

    list_of_files = glob.glob(str(model_filepath / '*.keras'))
    if not list_of_files: # add check 
        raise FileNotFoundError(f"No checkpoint files found in {model_filepath}")

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Latest checkpoint file: {latest_file}")  # Debugging statement
    return Path(latest_file)

def load_bk(BK_file:Path):
    BK_file = str(BK_file)
    if BK_file.endswith('bpmn'):
        bpmn = pm4py.read_bpmn(BK_file)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
        return {"net": net, "initial_marking": initial_marking, "final_marking": final_marking,"type":BK_type.Procedural}
    elif BK_file.endswith('pnml'):
        shared.BK_type = BK_type.Procedural
        net, initial_marking, final_marking = pm4py.read_pnml(BK_file)
        gviz = vis_factory.apply(net, initial_marking, final_marking)
        #vis_factory.view(gviz)
        return {"net": net, "initial_marking":initial_marking, "final_marking":final_marking,"type":BK_type.Procedural}
    elif BK_file.endswith('.decl'):
        declare_model = DeclareModel().parse_from_file(BK_file)
        model_constraints = declare_model.get_decl_model_constraints()
        for idx, constr in enumerate(model_constraints):
            print(idx, constr)
        return {"model": declare_model,"type":BK_type.Declare}
    elif BK_file.endswith('.txt'):
        probDeclarePredictor = ProbDeclarePredictor()
        probDeclarePredictor.loadProbDeclModel(BK_file)
        return {"model": probDeclarePredictor,"type":BK_type.ProbDeclare}
    else:
        raise ValueError(
            f"The BK model '{BK_file}' must be one of the following types:\n"
            "- .bpmn or .pnml for procedural background knowledge\n"
            "- .decl for (MP)-Declare background knowledge\n"
            "- .txt for probabilistic Declare background knowledge"
        )

def discover_Petri_nets(log_data: LogData,pn_folder:Path=shared.pn_folder):
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    tree = pm4py.discover_process_tree_inductive(evaluation_traces, noise_threshold = 0.0,  activity_key=log_data.act_name_key,
                                                            case_id_key=log_data.case_name_key,
                                                            timestamp_key= log_data.timestamp_key)
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    if not Path.exists(pn_folder):
        Path.mkdir(pn_folder, parents=True)
    bk_filename = pn_folder / (log_data.log_name.value + '.pnml')
    pm4py.write_pnml(net, initial_marking, final_marking, bk_filename)

def prepare_encoded_data(log_data: LogData, resource: bool):
    """
    Get all possible symbols for activities and resources and annotate them with integers.
    """
    act_name_key = log_data.act_name_key
    act_chars = log_data.log[act_name_key].unique().tolist()
    act_chars.sort()
    target_act_chars = copy.copy(act_chars)
    target_act_chars.append('!')

    act_to_int = dict((c, i+1) for i, c in enumerate(act_chars))
    target_act_to_int = dict((c, i+1) for i, c in enumerate(target_act_chars))
    target_int_to_act = dict((i+1, c) for i, c in enumerate(target_act_chars))

    if resource:
        res_name_key = log_data.res_name_key
        res_chars = list(log_data.log[res_name_key].unique())
        res_chars.sort()
        target_res_chars = copy.copy(res_chars)
        target_res_chars.append('!')
        res_to_int = dict((c, i+1) for i, c in enumerate(res_chars))
        target_res_to_int = dict((c, i+1) for i, c in enumerate(target_res_chars))
        target_int_to_res = dict((i+1, c) for i, c in enumerate(target_res_chars))
    else:
        res_chars = None
        res_to_int = None
        target_res_to_int = None
        target_int_to_res = None
    return act_chars, res_chars, act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res