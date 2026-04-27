from pathlib import Path

from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import extract_last_model_checkpoint, Encodings,\
    prepare_encoded_data, BK_type
from NeSy4PPM.Prediction import inference_algorithm
from NeSy4PPM.ProbDeclmonitor.probDeclPredictor import AggregationMethod


def predict_evaluate(log_data: LogData, models_folder:Path, encoder: Encodings, output_filename:str,
                     evaluation_trace_ids=None, bk_model=None,
                     beam_size=3, method_fitness: str=None, ProbAgg_method: AggregationMethod=None,
                     weight: float=0.0, resource: bool=False, bk_end:bool=False):
    if bk_model is not None and bk_end and bk_model["type"] == BK_type.Declare: bk_model["type"] = BK_type.Declare_End
    elif bk_model is not None and bk_end and bk_model["type"] == BK_type.Procedural: bk_model["type"] = BK_type.Procedural_End
    maxlen = log_data.max_len
    chars, chars_group, act_to_int, target_act_to_int, target_int_to_act,res_to_int, target_res_to_int, target_int_to_res \
        = prepare_encoded_data(log_data,resource)
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    if evaluation_trace_ids is not None:
        evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(evaluation_trace_ids)]
    prediction_type = 'CF' + 'R' * resource

    folder_path = models_folder / 'results21' / prediction_type
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    print(f"{'Activity' + ' & Resource'*resource} Prediction ...")
    output_filename = folder_path / output_filename
    model_filename = extract_last_model_checkpoint(log_data.log_name, 'CF' + 'R'*resource,models_folder)
    inference_algorithm.run_experiments(log_data, evaluation_traces, maxlen, encoder, act_to_int, target_int_to_act,
                                   res_to_int, target_int_to_res, model_filename, output_filename, bk_model,
                                   method_fitness,ProbAgg_method, resource, weight, bk_end,beam_size)