"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import itertools
from typing import Dict
import operator
import numpy as np
import pm4py
import pandas as pd
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.Utils.Declare.TraceStates import TraceState

from NeSy4PPM.Data_preprocessing import shared_variables as shared
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import Encodings, BK_type
from NeSy4PPM.Prediction.Checkers import TraceDeclareAnalyzer
from NeSy4PPM.Prediction.create_event_log import convert_to_log
from enum import Enum

class ConstraintChecker(Enum):
    SATISFIED = 1
    POSSIBLY_SATISFIED = 0.66
    POSSIBLY_VIOLATED = 0.33
    VIOLATED = 0


def get_pn_fitness(bk_model, method_fitness: str, log: pd.DataFrame, log_data: LogData, completed_trace:bool=True) -> Dict[str, float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s')
    net = bk_model["net"]
    initial_marking = bk_model["initial_marking"]
    final_marking = bk_model["final_marking"]

    acts = list(dec_log[log_data.act_name_key]) # list of prefix activities

    def replayed(a):
        return [t.label for t in a["activated_transitions"] if t.label is not None]

    if method_fitness == "fitness_alignments":
        alignments = pm4py.conformance_diagnostics_alignments(dec_log, net, initial_marking, final_marking,
                                                                  activity_key=log_data.act_name_key,
                                                                  case_id_key=log_data.case_name_key,
                                                                  timestamp_key=log_data.timestamp_key)
        trace_fitnesses = [a['fitness'] for a in alignments]
    elif method_fitness == "fitness_token_based_replay":
        alignments = pm4py.conformance_diagnostics_token_based_replay(dec_log, net, initial_marking, final_marking,
                                                                      activity_key=log_data.act_name_key,
                                                                      case_id_key=log_data.case_name_key,
                                                                      timestamp_key=log_data.timestamp_key)
        trace_fitnesses = (
            [1.0 if a["trace_is_fit"] else 0.0 for a in alignments] if completed_trace
            else [1.0 if (not a["transitions_with_problems"] # no missing transitions
                        and acts == replayed(a)) #all activities in the prefix are activated
                 else 0.0
                for  a in  alignments
            ])
    trace_ids = list(log[log_data.case_name_key].unique())
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness


# === Helper functions ===


def encode(crop_trace: pd.DataFrame, log_data: LogData, encoder:Encodings, maxlen: int, char_indices: Dict[str, int],
                      char_indices_group: Dict[str, int], resource: bool) -> np.ndarray:
    """
    encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    if resource:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        sentence_group = ''.join(crop_trace[log_data.res_name_key].tolist())
        chars_group = list(char_indices_group.keys())
        if encoder== Encodings.One_hot:
            num_features = len(chars) + len(chars_group)
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if char in chars:
                    x[0, t + leftpad, char_indices[char] - 1] = 1
                if t < len(sentence_group):
                    if sentence_group[t] in chars_group:
                        x[0, t + leftpad, len(char_indices) + char_indices_group[sentence_group[t]] - 1] = 1
        elif encoder== Encodings.Multi_encoders:
            num_features = maxlen
            x_a = np.zeros((1, num_features), dtype=np.float32)
            x_g = np.zeros((1, num_features), dtype=np.float32)
            for t, char in enumerate(sentence):
                x_a[0, t] = char_indices[char]
                x_g[0, t]= char_indices_group[sentence_group[t]]
            x = {
                'x_act': x_a,
                'x_group' : x_g
            }
        else:
            if encoder == Encodings.Shrunk_Index_based:
                result_list = [x + y for x, y in itertools.product(chars, chars_group)]
                target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
                num_features = maxlen
                x = np.zeros((1, num_features), dtype=np.float32)
                for t, char in enumerate(sentence):
                    x[0, t] = target_to_int[char + sentence_group[t]]
            else:
                num_features = maxlen * 2
                counter_act = 0
                counter_res = 1
                x = np.zeros((1, num_features), dtype=np.float32)
                for t, char in enumerate(sentence):
                    x[0, counter_act] = char_indices[char]
                    if t < len(sentence_group):
                        x[0, counter_res] = char_indices_group[sentence_group[t]]
                    counter_act += 2
                    counter_res += 2
    else:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        if encoder == Encodings.One_hot:
            num_features = len(chars)
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if char in chars:
                    x[0, t + leftpad, char_indices[char]-1] = 1
        else:
            num_features = maxlen
            x = np.zeros((1, num_features), dtype=np.float32)
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]
    return x

def get_beam_size(self, NodePrediction, current_child_node, bk_model, weight, prefix_trace,
                  prediction, res_prediction, target_ind_to_act, target_ind_to_res,
                  log_data, resource, beam_size,cache_fitness, method_fitness, ProbAgg_method):
    score=prediction
    act_prefix = prefix_trace.cropped_line
    prefix_trace = prefix_trace.cropped_trace if isinstance(prefix_trace, NodePrediction) else prefix_trace
    if resource:
        prob_matrix = np.log(prediction) + np.log(res_prediction[:, np.newaxis])
        if bk_model and bk_model["type"] == BK_type.Declare:
            for res_pred_idx, act_pred_idx in np.ndindex(prob_matrix.shape):
                temp_prediction = target_ind_to_act[act_pred_idx + 1]
                temp_res_prediction = target_ind_to_res[res_pred_idx + 1]
                BK_res = declare_compliance_checking(log_data, temp_prediction, temp_res_prediction, bk_model["model"],
                                                         prefix_trace, resource)
                prob_matrix[res_pred_idx][act_pred_idx] = (prob_matrix[res_pred_idx][act_pred_idx] * 0.5 * (1-weight)) + (BK_res * weight)
        sorted_prob_matrix = np.argsort(prob_matrix, axis=None)[::-1]
    else:
        if bk_model and bk_model["type"] == BK_type.Declare:
            BK_res = np.zeros(len(target_ind_to_act), dtype=np.float32)
            for f in range(1, len(target_ind_to_act) + 1):
                temp_prediction = target_ind_to_act[f]
                BK_res [f-1] = declare_compliance_checking(log_data, temp_prediction, None, bk_model["model"],
                                                                     prefix_trace, resource)
            score =BK_res if prediction is None else [(np.log(a) * (1-weight)) + (b * weight) for a, b in zip(prediction, BK_res)]
        elif bk_model and bk_model["type"] == BK_type.ProbDeclare:
            prefix_act = [log_data.act_enc_mapping[index] for index in act_prefix]
            results = bk_model["model"].processPrefix(prefix_act,ProbAgg_method)
            BK_res = np.zeros(len(target_ind_to_act), dtype=np.float32)
            target_acts = [log_data.act_enc_mapping[target_ind_to_act[indice]] if target_ind_to_act[indice] != '!'  # list of activities
                           else 'False'  # End symbol
                           for indice in range(1, len(target_ind_to_act) + 1)
                           ]
            for event, score in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
                if event is False:  # End symbol
                    BK_res[target_acts.index('False')] = round(score, 3)
                elif event is not True:  # Activities present in the declare model
                    BK_res[target_acts.index(event)] = round(score, 3)
                else:  # if event is True: Activities not present in the declare model
                    act_indices = [i for i, act in enumerate(target_acts) if act not in results.keys() and act != 'False']
                    for idx in act_indices:
                        BK_res[idx] = round(score, 3)
            score = [(np.log(a) * (1 - weight)) + (np.log(b) * weight) for a, b in zip(prediction, BK_res)]
        elif bk_model and bk_model["type"] == BK_type.Procedural:
            fitness_res = fitness_checking(log_data, method_fitness, target_ind_to_act ,cache_fitness, bk_model,prefix_trace)
            score = fitness_res if prediction is None else [(np.log(a) * (1 - weight)) + (np.log(b)* weight) for a, b in zip(prediction, fitness_res)]

    for j in range(beam_size):
        prefix_trace = prefix_trace.cropped_trace if isinstance(prefix_trace, NodePrediction) else prefix_trace

        if resource:
            res_pred_idx, act_pred_idx = np.unravel_index(sorted_prob_matrix[j], prob_matrix.shape)
            temp_prediction = target_ind_to_act[act_pred_idx + 1]
            temp_res_prediction = target_ind_to_res[res_pred_idx + 1]
            probability_this = prob_matrix[res_pred_idx][act_pred_idx]
        else:
            pred_idx = np.argsort(score)[len(score) - j - 1]
            temp_prediction = target_ind_to_act[pred_idx + 1]
            temp_res_prediction = None
            probability_this = np.sort(score)[len(score) - j - 1]

        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_prediction
        if resource: predicted_row.loc[:, log_data.res_name_key] = temp_res_prediction
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)

        probability_of = (current_child_node.probability_of + probability_this)

        temp = NodePrediction(temp_cropped_trace_next,probability_of)
        self.put(temp)
    return self

def declare_compliance_checking(log_data, bk_model, prefix_trace, temp_prediction:None, temp_res_prediction:None, resource=False):
    if temp_prediction == "!" or (resource and temp_res_prediction == "!"):
        completed_trace = True
        temp_cropped_trace_next = prefix_trace.copy()
    else:
        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_prediction
        if resource:
            predicted_row.loc[:, log_data.res_name_key] = temp_res_prediction
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)

    temp_cropped_trace_next[log_data.act_name_key] = temp_cropped_trace_next[log_data.act_name_key].apply(
        lambda x: x.replace(x, log_data.act_enc_mapping[x]))
    if resource:
        temp_cropped_trace_next[log_data.res_name_key] = temp_cropped_trace_next[log_data.res_name_key].apply(
        lambda x: x.replace(str(x), log_data.res_enc_mapping[x] if x != "!" else ""))
    log = convert_to_log(temp_cropped_trace_next, log_data.case_name_key, log_data.act_name_key)
    d_log = D4PyEventLog()
    d_log.log = log
    d_log.log_length = len(d_log.log)
    d_log.timestamp_key = log_data.timestamp_key
    d_log.activity_key = log_data.act_name_key
    basic_checker = TraceDeclareAnalyzer(log=d_log, declare_model=bk_model,
                                         consider_vacuity=True, completed=completed_trace)
    conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
    state = conf_check_res.get_metric(metric="state", trace_id=0)
    if 0 in state:
        BK_result = np.NINF  # violated constraint found
    else:
        results = []
        for result in conf_check_res.model_check_res[0]:
            if result.state == TraceState.POSSIBLY_SATISFIED.value:
                results.append(ConstraintChecker.POSSIBLY_SATISFIED.value)
            elif result.state == TraceState.SATISFIED.value:
                results.append(ConstraintChecker.SATISFIED.value)
            elif result.state == TraceState.POSSIBLY_VIOLATED.value:
                results.append(ConstraintChecker.POSSIBLY_VIOLATED.value)
        BK_result = np.log(np.mean(results))
    return BK_result

def fitness_checking(log_data, method_fitness, target_indices_char, cache_fitness, bk_file, prefix_trace):
    fitness = []
    for f in range(1, len(target_indices_char) + 1):
        temp_prediction = target_indices_char[f]
        if temp_prediction == "!":
            completed_trace = True
            temp_cropped_trace_next = prefix_trace.copy()
        else:
            completed_trace = False
            predicted_row = prefix_trace.tail(1).copy()
            predicted_row.loc[:, log_data.act_name_key] = temp_prediction
            temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row])
        trace_name = temp_cropped_trace_next[log_data.case_name_key].iloc[0]
        temp_cropped_line_next = ''.join(prefix_trace[log_data.act_name_key].tolist()+[temp_prediction])

        check_cache = cache_fitness.get(temp_cropped_line_next)
        if check_cache == None:
            fitness_current = get_pn_fitness(bk_file, method_fitness, temp_cropped_trace_next,
                                             log_data,completed_trace)[trace_name]
            cache_fitness.add(temp_cropped_line_next, fitness_current)
        else:
            fitness_current = check_cache

        fitness = fitness + [fitness_current]

    if np.all(fitness == fitness[0]):
        fitness = np.repeat(1 / len(fitness), len(fitness)).tolist()
    else:
        beta = 20
        max_f = np.max(fitness)
        fitness = [np.exp(beta * (f - max_f)) for f in fitness]
        fitness = [f / sum(fitness) for f in fitness]

    return fitness

