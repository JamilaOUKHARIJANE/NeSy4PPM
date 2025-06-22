import csv
from pathlib import Path
import distance
import keras
import numpy as np
import pandas as pd
from NeSy4PPM.commons import shared_variables as shared
from jellyfish import damerau_levenshtein_distance

from NeSy4PPM.commons.log_utils import LogData
from NeSy4PPM.prediction.prepare_data import encode
from NeSy4PPM.learning.train_common import CustomTransformer
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tqdm import tqdm

def run_experiments(log_data: LogData, compliant_traces: pd.DataFrame, maxlen, predict_size, char_indices,
                     target_indices_char, char_indices_group, target_indices_char_group, model_file: Path,
                    output_file: Path, method_fitness: str, resource: bool, weight: list):

    # Load model
    model = keras.models.load_model(model_file, custom_objects={'CustomTransformer': CustomTransformer})

    def apply_trace(trace, prefix_size, log_data, model, output_file, method_fitness, resource, outcome, weight):
        if len(trace) > prefix_size:
            trace_name = trace[log_data.case_name_key].iloc[0]
            trace_prefix = trace.head(prefix_size)
            act_prefix = ''.join(trace_prefix[log_data.act_name_key].tolist()) + "_" + str(weight)

            # Concatenate activities and resources in the trace prefix
            trace_prefix_act = ''.join(trace_prefix[log_data.act_name_key].tolist())
            trace_prefix_res = ''.join(trace_prefix[log_data.res_name_key].tolist()) if resource else None

            trace_ground_truth = trace.tail(trace.shape[0] - prefix_size)
            act_ground_truth = ''.join(trace_ground_truth[log_data.act_name_key].tolist())
            res_ground_truth = ''.join(trace_ground_truth[log_data.res_name_key].tolist()) if resource else None

            # Initial encoding for the prefix
            model_input = encode(trace_prefix, log_data, maxlen, char_indices, char_indices_group, resource)
            predicted_acts = []
            predicted_res = []

            cropped_line = ''.join(trace_prefix[log_data.act_name_key].tolist())
            cropped_line_group = ''.join(trace_prefix[log_data.res_name_key].tolist()) if resource else ''

            for i in range(predict_size - prefix_size):
                if shared.use_modulator:
                    y = model.predict([model_input["x_act"], model_input["x_group"]], verbose=0)
                else:
                    y = model.predict(model_input, verbose=0)

                if resource:
                    y_char = y[0][0]
                    y_group = y[1][0]
                else:
                    y_char=y
                    y_group=None

                next_act = target_indices_char[np.argmax(y_char) + 1]

                if resource:                    
                    next_res = target_indices_char_group[np.argmax(y_group) + 1]
                    if next_act == "!" or next_res=="!":
                        break
                    predicted_res.append(next_res)
                    predicted_acts.append(next_act)

                    next_char_df = pd.DataFrame([{log_data.act_name_key: next_act, log_data.res_name_key: next_res}])
                    cropped_line_group += next_res
                else:
                    if next_act == "!":
                        break
                    predicted_acts.append(next_act)
                    next_char_df = pd.DataFrame([{log_data.act_name_key: next_act}])

                trace_prefix = pd.concat([trace_prefix, next_char_df], ignore_index=True)
                cropped_line += next_act

                # Update the model input by encoding the new sequence with the predicted character appended
                model_input = encode(trace_prefix, log_data, maxlen, char_indices, char_indices_group, resource)


            predicted_acts_str = ''.join(predicted_acts)
            predicted_res_str = ''.join(predicted_res) if resource else None

            dls_acts = 1 - (damerau_levenshtein_distance(predicted_acts_str, act_ground_truth) / max(len(predicted_acts_str), len(act_ground_truth)))
            if dls_acts < 0:
                dls_acts = 0
            jaccard_acts = 1 - distance.jaccard(predicted_acts, act_ground_truth)

            if resource:
                dls_res = 1 - (damerau_levenshtein_distance(predicted_res_str, res_ground_truth) / max(len(predicted_res_str), len(res_ground_truth)))
                if dls_res < 0:
                    dls_res = 0
                jaccard_res = 1 - distance.jaccard(predicted_res, res_ground_truth)

                # Combine activity and resource strings for combined prediction
                combined_ground_truth = ''.join([a + r for a, r in zip(act_ground_truth, res_ground_truth)])
                combined_predicted = ''.join([a + r for a, r in zip(predicted_acts_str, predicted_res_str)])

                dls_combined = 1 - (damerau_levenshtein_distance(combined_predicted, combined_ground_truth) / max(len(combined_predicted), len(combined_ground_truth)))
                if dls_combined < 0:
                    dls_combined = 0

            output = [trace_name, prefix_size, trace_prefix_act, act_ground_truth, predicted_acts_str, dls_acts, jaccard_acts]
            if resource:
                output.extend([trace_prefix_res, res_ground_truth, predicted_res_str, dls_res, jaccard_res, dls_combined])
            output.extend([weight])

            with open(output_file, 'a', encoding='utf-8', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(output)

    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        if resource:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["Case Id", "Prefix length", "Trace Prefix Act", "Ground truth",
                                 "Predicted Acts", "Damerau-Levenshtein Acts", "Jaccard Acts",
                                 "Trace Prefix Res" ,"Ground truth Resources", "Predicted Resources",
                                 "Damerau-Levenshtein Resources", "Jaccard Resources",
                                 "Damerau-Levenshtein Combined", "Weight"])
        else:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["Case Id", "Prefix length", "Trace Prefix Act", "Ground truth", "Predicted",
                                 "Damerau-Levenshtein", "Jaccard", "Weight"])

    for prefix_size in range(log_data.evaluation_prefix_start, log_data.evaluation_prefix_end + 1):
        print(prefix_size)
        compliant_traces = compliant_traces.reset_index(drop=True)
        for w in weight:
            tqdm.pandas()
            compliant_traces.groupby(log_data.case_name_key).progress_apply(lambda x: apply_trace(x, prefix_size, log_data,
                                                                                    model, output_file,  
                                                                                    method_fitness, resource, w))