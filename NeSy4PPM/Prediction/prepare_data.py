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
import torch
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.Utils.Declare.TraceStates import TraceState

from NeSy4PPM.Data_preprocessing import shared_variables as shared
from NeSy4PPM.Data_preprocessing.utils import Encodings, BK_type
from NeSy4PPM.Prediction.Checkers import TraceDeclareAnalyzer
from NeSy4PPM.Prediction.create_event_log import convert_to_log
from enum import Enum

class ConstraintChecker(Enum):
    SATISFIED = 1
    POSSIBLY_SATISFIED = 0.66
    POSSIBLY_VIOLATED = 0.33
    VIOLATED = 0


def get_pn_fitness(bk_model, method_fitness: str, log: pd.DataFrame, log_data, completed_trace:bool=True) -> Dict[str, float]:
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
def encode(crop_trace: pd.DataFrame, log_data, encoder:Encodings, maxlen: int, char_indices: Dict[str, int],
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

def compute_score(prefix_trace, child_node, log_data, bk_model, weight, resource=False, feasibility_pruning=False,
                  end_distance_pruning=False, ProbAgg_method=None, method_fitness=None, cache_fitness=None):
    prediction = None
    SDFA_states = None
    # Compute NN score.
    if weight != 1.0:
        prediction= child_node.nn_probability

    # Compute BK score.
    BK_res = None
    if resource:
        if bk_model and bk_model["type"] == BK_type.Declare:
            BK_res = np.zeros((len(log_data.target_int_to_act), len(log_data.target_int_to_res)), dtype=np.float32)
            for act_pred_idx, res_pred_idx in np.ndindex(BK_res.shape):
                temp_next_act = log_data.target_int_to_act[act_pred_idx + 1]
                temp_next_res = log_data.target_int_to_res[res_pred_idx + 1]
                BK_res[act_pred_idx, res_pred_idx] = declare_compliance_checking(log_data, bk_model["model"],
                                                                                 prefix_trace, temp_next_act,
                                                                                 temp_next_res, resource)
    else:
        if bk_model and bk_model["type"] == BK_type.Declare:
            BK_res = np.zeros(len(log_data.target_int_to_act), dtype=np.float32)
            for idx, temp_next_act in log_data.target_int_to_act.items():
                BK_res[idx - 1] = declare_compliance_checking(log_data, bk_model["model"], prefix_trace, temp_next_act)
        if bk_model and bk_model["type"] == BK_type.ProbDeclare:
            act_prefix = child_node.cropped_line
            prefix_act = [log_data.act_enc_mapping[index] for index in act_prefix]
            results = bk_model["model"].processPrefix(prefix_act, ProbAgg_method)
            BK_res = np.zeros(len(log_data.target_int_to_act), dtype=np.float32)
            target_acts = [log_data.act_enc_mapping[target_act] if target_act != "!" else "False"
                for target_act in log_data.target_int_to_act.values()]
            for event, event_score in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
                if event is False:  # End symbol
                    BK_res[target_acts.index("False")] = round(event_score, 3)
                elif event is not True:  # Activities present in the declare model
                    BK_res[target_acts.index(event)] = round(event_score, 3)
                else:  # if event is True: Activities not present in the declare model
                    act_indices = [i for i, act in enumerate(target_acts) if act not in results.keys() and act != "False"]
                    for idx in act_indices:
                        BK_res[idx] = round(event_score, 3)
        if bk_model and bk_model["type"] == BK_type.Procedural:
            BK_res = np.asarray(
                fitness_checking(log_data, method_fitness, log_data.target_int_to_act, cache_fitness, bk_model, prefix_trace),
                dtype=np.float32,
            )
        if bk_model and bk_model["type"] == BK_type.SDFA:
            sdfa_model = bk_model["model"]
            BK_res = np.zeros(len(log_data.target_int_to_act), dtype=np.float32)
            R = log_data.max_len - (len(prefix_trace) + 1)
            SDFA_states = {}
            for idx, temp_next_act in log_data.target_int_to_act.items():
                if temp_next_act != "!":
                    encoded_act = log_data.act_enc_mapping[temp_next_act]
                    next_SDFA_state = sdfa_model.next_state(child_node.SDFA_state, encoded_act)
                    SDFA_prob = sdfa_model.probability(child_node.SDFA_state, encoded_act)
                    if feasibility_pruning:
                        if next_SDFA_state is None or SDFA_prob == 0:
                            continue
                    else:
                        if next_SDFA_state is None:
                            SDFA_prob = 0
                            next_SDFA_state = child_node.SDFA_state

                    if end_distance_pruning :
                        termination_distance = sdfa_model.termination_distance(next_SDFA_state)
                        if termination_distance > R:
                            continue
                    BK_res[idx - 1] = SDFA_prob
                else:  # End symbol
                    next_SDFA_state = child_node.SDFA_state
                    end_symb_prob = sdfa_model.end_probability(child_node.SDFA_state)
                    if feasibility_pruning and end_symb_prob == 0:
                        continue
                    BK_res[idx - 1] = end_symb_prob
                SDFA_states[idx - 1] = next_SDFA_state

    # Final score
    if prediction is None and BK_res is None:
        raise ValueError("Cannot compute score without NN prediction or BK result.")
    if prediction is None:
        score = np.log(BK_res)
    elif BK_res is None:
        score = np.log(prediction)
    else:
        score = (np.log(prediction) * (1 - weight)) + (np.log(BK_res) * weight)

    return score, SDFA_states

class _DynamicValue:
    def __init__(self, value_getter):
        self._value_getter = value_getter

    @property
    def value(self):
        return self._value_getter()

def get_beam_size(self, NodePrediction, current_node, bk_model, weight, log_data, resource, beam_size, cache_fitness,
                  method_fitness, ProbAgg_method):
    SDFA_state = current_node.SDFA_state
    prefix_trace = current_node.cropped_trace
    end_distance_pruning = _DynamicValue(lambda:shared.end_distance_pruning).value
    feasibility_pruning= _DynamicValue(lambda: shared.hard_pruning).value
    score, SDFA_states = compute_score(prefix_trace, current_node, log_data, bk_model, weight, resource=resource,
                                       ProbAgg_method=ProbAgg_method, method_fitness=method_fitness,feasibility_pruning=feasibility_pruning,
                                       end_distance_pruning=end_distance_pruning,
                                       cache_fitness=cache_fitness)

    if feasibility_pruning or end_distance_pruning:
        valid_indices = np.flatnonzero(np.isfinite(score))
        valid_indices = valid_indices[np.argsort(score.flat[valid_indices])[::-1]]
    else:
        valid_indices = np.argsort(score, axis=None)[::-1]

    for candidate_idx in valid_indices[:beam_size or len(valid_indices)]:
        if resource:
            act_pred_idx, res_pred_idx = np.unravel_index(candidate_idx, score.shape)
            temp_next_act = log_data.target_int_to_act[act_pred_idx + 1]
            temp_next_res = log_data.target_int_to_res[res_pred_idx + 1]
            probability_this = score[act_pred_idx, res_pred_idx]
        else:
            pred_idx = candidate_idx
            temp_next_act = log_data.target_int_to_act[pred_idx + 1]
            temp_next_res = None
            probability_this = score[pred_idx]
            if bk_model and bk_model["type"] == BK_type.SDFA and weight != 0 and temp_next_act != "!": SDFA_state = SDFA_states[pred_idx]

        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_next_act
        if resource: predicted_row.loc[:, log_data.res_name_key] = temp_next_res
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)

        probability_of = (current_node.probability_of + probability_this)

        temp = NodePrediction(temp_cropped_trace_next, probability_of, SDFA_state)
        self.put(temp)
    return self

def declare_compliance_checking(log_data, bk_model, prefix_trace, temp_next_act= "!", temp_next_res= "!", resource=False):
    completed_trace = False
    if temp_next_act == "!" or (resource and temp_next_res == "!"):
        completed_trace = True
        temp_cropped_trace_next = prefix_trace.copy()
    else:
        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_next_act
        if resource:
            predicted_row.loc[:, log_data.res_name_key] = temp_next_res
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
        BK_result = 1e-20  # violated constraint found
    else:
        results = []
        for result in conf_check_res.model_check_res[0]:
            if result.state == TraceState.POSSIBLY_SATISFIED.value:
                results.append(ConstraintChecker.POSSIBLY_SATISFIED.value)
            elif result.state == TraceState.SATISFIED.value:
                results.append(ConstraintChecker.SATISFIED.value)
            elif result.state == TraceState.POSSIBLY_VIOLATED.value:
                results.append(ConstraintChecker.POSSIBLY_VIOLATED.value)
        BK_result = np.mean(results)
    return BK_result

def fitness_checking(log_data, method_fitness, target_indices_char, cache_fitness, bk_file, prefix_trace):
    fitness = []
    for f in range(1, len(target_indices_char) + 1):
        temp_next_act = target_indices_char[f]
        if temp_next_act == "!":
            completed_trace = True
            temp_cropped_trace_next = prefix_trace.copy()
        else:
            completed_trace = False
            predicted_row = prefix_trace.tail(1).copy()
            predicted_row.loc[:, log_data.act_name_key] = temp_next_act
            temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row])
        trace_name = temp_cropped_trace_next[log_data.case_name_key].iloc[0]
        temp_cropped_line_next = ''.join(prefix_trace[log_data.act_name_key].tolist()+[temp_next_act])

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
