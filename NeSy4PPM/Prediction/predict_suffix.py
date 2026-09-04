from pathlib import Path

from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import extract_last_model_checkpoint, Encodings, BK_type
from NeSy4PPM.Prediction import inference_algorithm
from NeSy4PPM.ProbDeclmonitor.probDeclPredictor import AggregationMethod


def predict_evaluate(log_data: LogData, models_folder:Path, encoder: Encodings, output_filename:str,
                     evaluation_trace_ids=None, bk_model=None,
                     beam_size:int=3, method_fitness = None, ProbAgg_method: AggregationMethod=None,
                     weight: float=0.0, resource: bool=False, bk_end:bool=False, prefix_size:int=0):
    if bk_model is not None and bk_end and bk_model["type"] == BK_type.Declare: bk_model["type"] = BK_type.Declare_END
    elif bk_model is not None and bk_end and bk_model["type"] == BK_type.Procedural: bk_model["type"] = BK_type.Procedural_END
    elif bk_model is not None and bk_end and bk_model["type"] == BK_type.SDFA:
        bk_model["type"] = BK_type.SDFA_END

    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    if evaluation_trace_ids is not None:
        evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(evaluation_trace_ids)]
    prediction_type = 'CF' + 'R' * resource

    folder_path = models_folder / 'results' / prediction_type
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    print(f"{'Activity' + ' & Resource'*resource} Prediction ...")
    output_filename = folder_path / output_filename
    model_filename = extract_last_model_checkpoint(log_data.log_name, 'CF' + 'R'*resource,models_folder)
    inference_algorithm.run_experiments(log_data, evaluation_traces, encoder,model_filename, output_filename, bk_model,
                                   method_fitness,ProbAgg_method, resource, weight, bk_end, beam_size, prefix_size)