import time

import keras
import pm4py
import itertools
from pathlib import Path

from NeSy4PPM.commons.log_utils import LogData
from NeSy4PPM.commons import shared_variables as shared
from NeSy4PPM.commons.utils import extract_last_model_checkpoint, Encodings,\
    prepare_encoded_data, NN_model
from NeSy4PPM.prediction.inference_algorithms import beamsearch

def predict_evaluate(log_data: LogData, model_arch:NN_model, encoder: Encodings,evaluation_trace_ids=None,
                 output_folder:Path=shared.output_folder,bk_model=None,
                 beam_size=3, method_fitness: str=None,
                 weight: list=[0.0], resource: bool=False, bk_end:bool=False):
    start_time = time.time()
    shared.beam_size = beam_size
    maxlen = log_data.max_len
    chars, chars_group, act_to_int, target_act_to_int, target_int_to_act,res_to_int, target_res_to_int, target_int_to_res \
        = prepare_encoded_data(log_data,resource)
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    if evaluation_trace_ids is not None:
        evaluation_traces = evaluation_traces[evaluation_traces[log_data.case_name_key].isin(evaluation_trace_ids)]
    models_folder = model_arch.value + '_' + encoder.value
    for fold in range(shared.folds):
        prediction_type = 'CF' + 'R' * resource
        folder_path = output_folder / models_folder / str(fold) / 'results' / prediction_type
        if not Path.exists(folder_path):
            Path.mkdir(folder_path, parents=True)
        print(f"fold {fold} - {'Activity' + ' & Resource'*resource} Prediction")
        output_filename = folder_path / (f'{log_data.log_name if log_data.test_log_name is None else log_data.test_log_name}'
                                             f'_beam{str(shared.beam_size)}_fold{str(fold)}_cluster'
                                             f'{log_data.evaluation_prefix_start}_{shared.BK_type if shared.BK_type else ""}.csv')

        model_filename = extract_last_model_checkpoint(log_data.log_name, models_folder, fold, 'CF' + 'R'*resource,output_folder)
        beamsearch.run_experiments(log_data, evaluation_traces, maxlen, encoder, act_to_int, target_int_to_act,
                                   res_to_int, target_int_to_res, model_filename, output_filename, bk_model,
                                   method_fitness, resource, weight, bk_end)


        print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))