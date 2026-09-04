from __future__ import division
import csv
import math
import os
import time
from pathlib import Path
from queue import PriorityQueue
import distance
import numpy as np
import pandas as pd
import keras
from jellyfish import damerau_levenshtein_distance
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import Encodings, BK_type
from NeSy4PPM.Prediction.prepare_data import get_beam_size, encode, get_pn_fitness, declare_compliance_checking
from NeSy4PPM.Training.train_common import get_keras_custom_objects
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tqdm import tqdm


def load_trained_model(model_file: Path):
    model_file = Path(model_file)
    return keras.models.load_model(model_file, custom_objects=get_keras_custom_objects())

def run_experiments(log_data: LogData, evaluation_traces: pd.DataFrame, encoder: Encodings, model_file: Path,
                    output_file: Path, bk_model,
                    method_fitness = None, ProbAgg_method=None, resource: bool = False, weight: float = 0.0,
                    bk_end: bool = False, beam_size: int = 1, prefix_size:int=0):
    maxlen = log_data.max_len
    model = load_trained_model(model_file)

    class NodePrediction:
        def __init__(self, crop_trace: pd.DataFrame, probability_of=0, SDFA_state=None):
            self.cropped_trace = crop_trace
            self.cropped_line = ''.join(crop_trace[log_data.act_name_key].tolist())
            if resource:
                self.cropped_line_group = ''.join(crop_trace[log_data.res_name_key].tolist())

            line_ended = bool(self.cropped_line) and self.cropped_line[-1] == "!"  # not end symbol
            group_ended = resource and bool(self.cropped_line_group) and self.cropped_line_group[
                -1] == "!"  # not last resource end symbol
            ended = line_ended or group_ended

            full = len(self.cropped_line) > maxlen  # prefix length must be <= maxlen
            if not ended and not full and weight != 1.0:
                if str(model_file).endswith(".keras"):
                    enc = encode(crop_trace, log_data, encoder, maxlen, log_data.act_to_int,
                                              log_data.res_to_int, resource)
                    if encoder == Encodings.Multi_encoders:
                        y = model.predict([enc["x_act"], enc["x_group"]], verbose=0)
                    else:
                        y = model.predict(enc, verbose=0)
                    if resource:
                        self.nn_probability = (y[0][0][:, np.newaxis] + y[1][0][np.newaxis, :]) / 2
                    else:
                        self.nn_probability = y[0]
            self.probability_of = probability_of
            self.SDFA_state = SDFA_state

        def __str__(self):
            return f"Prefix: {self.cropped_line}, prob. {self.probability_of}, SDFA_state: {self.SDFA_state}"

        def __lt__(self, other):
            return -self.probability_of < -other.probability_of

        def get_cropped_trace(self):
            return self.cropped_trace

    class CacheFitness:
        def __init__(self):
            self.trace = {}

        def add(self, crop_trace: str, fitness: float):
            self.trace[crop_trace] = fitness

        def get(self, crop_trace: str):
            if crop_trace not in self.trace.keys():
                return None
            else:
                return self.trace[crop_trace]

    class CacheTrace:
        def __init__(self):
            self.trace = {}

        def add(self, crop_trace: str, output: list):
            self.trace[crop_trace] = output

        def get(self, crop_trace: str):
            if crop_trace not in self.trace.keys():
                return None
            else:
                return self.trace[crop_trace]

    def check_bk_end(child_node, log_data, bk_model, method_fitness, resource):
        bk_score = None
        prefix_trace = child_node.cropped_trace
        prefix_trace = prefix_trace[:-1]
        trace_name = prefix_trace[log_data.case_name_key].iloc[0]
        if resource and bk_model["type"] == BK_type.Declare_END:
            compliance = declare_compliance_checking(log_data, bk_model["model"], prefix_trace, resource=resource)
            bk_score = -np.inf if compliance == 1e-20 else compliance
        else:
            if bk_model["type"] == BK_type.Declare_END:
                compliance = declare_compliance_checking(log_data, bk_model["model"], prefix_trace)
                bk_score = -np.inf if compliance == 1e-20 else compliance
            if bk_model["type"] == BK_type.Procedural_END:
                fitness = get_pn_fitness(bk_model, method_fitness, prefix_trace, log_data)[trace_name]
                bk_score = -np.inf if fitness < 1.0 else fitness
            if bk_model["type"] == BK_type.SDFA_END:
                activities = (
                    log_data.act_enc_mapping[i]
                    for i in prefix_trace[log_data.act_name_key].tolist()
                )
                compliance = bk_model["model"].is_compliant_trace(activities)
                bk_score = 1.0 if compliance else -np.inf
        return bk_score

    def get_initial_sdfa_state(prefix_trace):
        if bk_model and bk_model["type"] == BK_type.SDFA:
            return bk_model["model"].replay(
                log_data.act_enc_mapping[i] for i in prefix_trace[log_data.act_name_key].tolist())
        return None

    def check_SDFA_end(child_node, bk_model):
        if bk_model and bk_model["type"] == BK_type.SDFA and weight != 0.0:
            return bk_model["model"].end_probability(child_node.SDFA_state) > 0
        return True

    def get_predicted_suffix(final_node, prefix_size, resource):
        act_end = -1 if final_node.cropped_line.endswith("!") else None
        predicted = final_node.cropped_line[prefix_size:act_end]

        if resource:
            group_end = -1 if final_node.cropped_line_group.endswith("!") else None
            predicted_group = final_node.cropped_line_group[prefix_size:group_end]
            return predicted, predicted_group
        return predicted, None

    def apply_trace(trace, prefix_size, log_data, predict_size, bk_model, method_fitness, resource, weight, bk_end, beam_size):

        if len(trace) > prefix_size:
            trace_name = trace[log_data.case_name_key].iloc[0]
            trace_prefix = trace.head(prefix_size)

            # Concatenate activities and resources in the trace prefix
            trace_prefix_act = ''.join(trace_prefix[log_data.act_name_key].tolist())
            act_prefix = ''.join(trace_prefix[log_data.act_name_key].tolist()) + "_" + str(weight)

            if resource:
                trace_prefix_res = ''.join(trace_prefix[log_data.res_name_key].tolist())
                res_prefix = ''.join(trace_prefix[log_data.res_name_key].tolist())

            check_prefix = cache_trace.get(act_prefix + "" + res_prefix) if resource else cache_trace.get(act_prefix)
            if check_prefix is None:
                trace_ground_truth = trace.tail(trace.shape[0] - prefix_size)
                act_ground_truth = ''.join(trace_ground_truth[log_data.act_name_key].tolist())

                if resource:
                    res_ground_truth = ''.join(trace_ground_truth[log_data.res_name_key].tolist())

                # Initialize queue for beam search, put root of the tree inside
                visited_nodes: PriorityQueue[NodePrediction] = PriorityQueue()
                visited_nodes.put(NodePrediction(trace_prefix, 0, SDFA_state=get_initial_sdfa_state(trace_prefix)))
                frontier_nodes: PriorityQueue[NodePrediction] = PriorityQueue()

                stop_expansion = False
                start_time = time.time()
                completed_nodes = []
                first_completed = None
                best_current_node = visited_nodes.queue[0] if visited_nodes.queue else None
                ebc_total = 0
                for step in range((predict_size - prefix_size) + 2):
                    if visited_nodes.empty():
                        break
                    best_current_node = visited_nodes.queue[0]
                    num_candidates = min(beam_size, len(visited_nodes.queue)) if beam_size else len(visited_nodes.queue)
                    for k in range(num_candidates):
                        current_node = visited_nodes.get()
                        if current_node.cropped_line[-1] == "!" or (resource and current_node.cropped_line_group[-1] == "!"):
                            if k == 0 and not bk_end:
                                if check_SDFA_end(current_node, bk_model):
                                    stop_expansion = True
                                    break
                                else:
                                    continue
                            else:
                                if bk_end:
                                    if first_completed is None and k == 0:
                                        first_completed = current_node
                                    bk_score = check_bk_end(current_node, log_data, bk_model, method_fitness, resource)
                                    completed_nodes.append((current_node, bk_score))
                                continue
                        if step > (predict_size - prefix_size):
                            stop_expansion = True
                        else:
                            # (Local: for each branch): Get the best `beam_size` children for the current node.
                            frontier_nodes = get_beam_size(frontier_nodes, NodePrediction, current_node,
                                                           bk_model, weight, log_data, resource, beam_size,
                                                           cache_fitness, method_fitness, ProbAgg_method )
                    if stop_expansion:
                        break
                    # (Global: for the entire beam search): Keep only the best `beam_size` children
                    explored_branch_count = len(frontier_nodes.queue)
                    ebc_total += explored_branch_count
                    visited_nodes = PriorityQueue()
                    nb_branches = min(beam_size, len(frontier_nodes.queue)) if beam_size else len(frontier_nodes.queue)
                    for _ in range(nb_branches):
                        visited_nodes.put(frontier_nodes.get())
                    if not visited_nodes.empty():
                        best_current_node = visited_nodes.queue[0]
                    frontier_nodes = PriorityQueue()
                final_node = None
                if bk_end:
                    compliant_completed = [x for x in completed_nodes if x[1] != -np.inf]
                    if compliant_completed:
                        final_node = first_completed if first_completed in compliant_completed else \
                        max(compliant_completed, key=lambda x: x[0].probability_of)[0]
                    elif first_completed is not None:
                        final_node = first_completed
                if final_node is None:
                    final_node = best_current_node  # best unfinished node or first completed node for approaches that not use BK at the end of prediction
                predicted, predicted_group = get_predicted_suffix(final_node, prefix_size, resource)

                output = []
                if len(act_ground_truth) > 0:
                    prediction_time = time.time() - start_time
                    output.append(trace_name)
                    output.append(prefix_size)
                    output.append(
                        '>>'.join([log_data.act_enc_mapping[i] if i != "!" else "" for i in trace_prefix_act]))
                    output.append(
                        '>>'.join([log_data.act_enc_mapping[i] if i != "!" else "" for i in act_ground_truth]))
                    output.append('>>'.join([log_data.act_enc_mapping[i] if i != "!" else "" for i in predicted]))
                    dls = 1 - \
                          (damerau_levenshtein_distance(predicted, act_ground_truth) / max(len(predicted),
                                                                                           len(act_ground_truth)))
                    if dls < 0:
                        dls = 0
                    output.append(dls)
                    output.append(1 - distance.jaccard(predicted, act_ground_truth))

                    if resource:
                        output.append(
                            '>>'.join([log_data.res_enc_mapping[i] if i != "!" else "" for i in trace_prefix_res]))
                        output.append(
                            '>>'.join([log_data.res_enc_mapping[i] if i != "!" else "" for i in res_ground_truth]))
                        output.append(
                            '>>'.join([log_data.res_enc_mapping[i] if i != "!" else "" for i in predicted_group]))
                        dls_res = 1 - \
                                  (damerau_levenshtein_distance(predicted_group, res_ground_truth)
                                   / max(len(predicted_group), len(res_ground_truth)))
                        if dls_res < 0:
                            dls_res = 0
                        output.append(dls_res)
                        output.append(1 - distance.jaccard(predicted_group, res_ground_truth))
                    output.append(weight)
                    output.append(prediction_time)
                    output.append(ebc_total)
                    output_cache = output.copy()
                    if resource: output_cache.append(predicted_group)
                    output_cache.append(predicted)
                    cache_trace.add(act_prefix + "" + res_prefix, output_cache) if resource else cache_trace.add(
                        act_prefix, output_cache)
                    print(output)
            else:
                trace_ground_truth = trace.tail(trace.shape[0] - prefix_size)
                act_ground_truth = ''.join(trace_ground_truth[log_data.act_name_key].tolist())
                output = []

                output.append(trace_name)
                output.append(prefix_size)
                output.append(check_prefix[2])  # Prefix
                output.append('>>'.join(
                    [log_data.act_enc_mapping[i] if i != "!" else "" for i in act_ground_truth]))  # Ground_truth
                predicted = check_prefix[-1]  # Predicted acts symbols
                output.append(check_prefix[4])  # Predicted acts labels
                dls = 1 - \
                      (damerau_levenshtein_distance(predicted, act_ground_truth) / max(len(predicted),
                                                                                       len(act_ground_truth)))
                if dls < 0:
                    dls = 0
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, act_ground_truth))
                if resource:
                    trace_prefix_res = ''.join(trace_prefix[log_data.res_name_key].tolist())
                    res_ground_truth = ''.join(trace_ground_truth[log_data.res_name_key].tolist())
                    predicted_group = check_prefix[-2]
                    output.append(
                        '>>'.join([log_data.res_enc_mapping[i] if i != "!" else "" for i in trace_prefix_res]))
                    output.append(
                        '>>'.join([log_data.res_enc_mapping[i] if i != "!" else "" for i in res_ground_truth]))
                    output.append(check_prefix[9])  # Predicted res labels

                    dls_res = 1 - (damerau_levenshtein_distance(predicted_group, res_ground_truth) / max(
                        len(predicted_group), len(res_ground_truth)))
                    dls_res = max(dls_res, 0)  # Ensure non-negative
                    output.append(dls_res)
                    output.append(1 - distance.jaccard(predicted_group, res_ground_truth))
                output.append(check_prefix[12]) if resource else output.append(check_prefix[7])  # weight
                output.append(check_prefix[13]) if resource else output.append(check_prefix[8])  # Prediction time
                output.append(check_prefix[14]) if resource else output.append(check_prefix[9])  # EBC
                print(output)

            if output:
                with open(output_file, 'a', encoding='utf-8', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(output)

    ##############################################################
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a+', encoding='utf-8', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            # Headers for the new file
            if resource:
                headers = ["Case ID", "Prefix length",
                           "Trace Prefix Act", "Ground truth", "Predicted Acts", "Damerau-Levenshtein Acts",
                           "Jaccard Acts",
                           "Trace Prefix Res", "Ground Truth Resources", "Predicted Resources",
                           "Damerau-Levenshtein Resources",
                           "Jaccard Resources", "Weight", "Time", "EBC"]
            else:
                headers = ["Case ID", "Prefix length", "Trace Prefix Act", "Ground truth", "Predicted Acts",
                           "Damerau-Levenshtein Acts", "Jaccard Acts", "Weight", "Time", "EBC"]
            spamwriter.writerow(headers)

    cache_fitness = CacheFitness()
    cache_trace = CacheTrace()
    if prefix_size > 0:
        evaluation_traces = evaluation_traces.reset_index(drop=True)
        tqdm.pandas()
        evaluation_traces.groupby(log_data.case_name_key, group_keys=False).progress_apply(lambda x: apply_trace(x,
                                                                                                                 prefix_size,
                                                                                                                 log_data,
                                                                                                                 maxlen,
                                                                                                                 bk_model,
                                                                                                                 method_fitness,
                                                                                                                 resource,
                                                                                                                 weight,
                                                                                                                 bk_end,
                                                                                                                 beam_size))
    else:
        prefix_percentages = [0.1, 0.25, 0.5, 0.75, 0.9]
        for percentage in prefix_percentages:  # prefix_size in range(log_data.evaluation_prefix_start, log_data.evaluation_prefix_end+1):
            evaluation_traces = evaluation_traces.reset_index(drop=True)
            tqdm.pandas()
            evaluation_traces.groupby(log_data.case_name_key, group_keys=False).progress_apply(lambda x: apply_trace(x,
                                                                                                                     max(1,
                                                                                                                         min(math.ceil(
                                                                                                                             len(x) * percentage),
                                                                                                                             len(x) - 1)),
                                                                                                                     log_data,
                                                                                                                     maxlen,
                                                                                                                     bk_model,
                                                                                                                     method_fitness,
                                                                                                                     resource,
                                                                                                                     weight,
                                                                                                                     bk_end,
                                                                                                                     beam_size))
