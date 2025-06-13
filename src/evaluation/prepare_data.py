"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import copy
import itertools
import re
from pathlib import Path
from typing import Dict
import operator
import numpy as np
import pm4py
import pandas as pd
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.Utils.Declare.TraceStates import TraceState

from src.commons import shared_variables as shared
from src.commons.log_utils import LogData
from src.evaluation.Checkers import TraceDeclareAnalyzer
from src.evaluation.create_event_log import convert_to_log
from enum import Enum

class ConstraintChecker(Enum):
    SATISFIED = 1
    POSSIBLY_SATISFIED = 0.66
    POSSIBLY_VIOLATED = 0.33
    VIOLATED = 0


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


def select_petrinet_compliant_traces(log_data: LogData,  method_fitness: str, traces: pd.DataFrame, path_to_pn_model_file: Path):
    """
    Select traces compliant to a Petri Net at least in a certain percentage specified as compliance_th
    """

    compliant_trace_ids = []
    if (method_fitness == "fitness_alignments") or  (method_fitness == "conformance_diagnostics_alignments_prefix"):
        method_fitness = "conformance_diagnostics_alignments"
    elif method_fitness == "fitness_token_based_replay":
        method_fitness = "conformance_diagnostics_token_based_replay"

    for trace_id, fitness in get_pn_fitness(path_to_pn_model_file, method_fitness, traces, log_data).items():
        if fitness >= log_data.compliance_th:
            compliant_trace_ids.append(trace_id)

    compliant_traces = traces[traces[log_data.case_name_key].isin(compliant_trace_ids)]
    return compliant_traces


def get_pn_fitness(bk_file: Path, method_fitness: str, log: pd.DataFrame, log_data: LogData) -> Dict[str, float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    #dec_
    log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s')

    if 'bpmn' in str(bk_file):
        bpmn = pm4py.read_bpmn(str(bk_file))
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
    else:
        evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(evaluation_traces,  activity_key=log_data.act_name_key,
                                                        case_id_key=log_data.case_name_key,
                                                        timestamp_key= log_data.timestamp_key)
        #.read_pnml(str(bk_file))

    if method_fitness == "conformance_diagnostics_alignments":
        alignments = pm4py.conformance_diagnostics_alignments(log,  net, initial_marking, final_marking,
                                                            activity_key=log_data.act_name_key,
                                                            case_id_key=log_data.case_name_key,
                                                            timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [a['fitness'] for a in alignments]
        
    elif method_fitness == "fitness_alignments":
        alignments = pm4py.fitness_alignments(log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
                                                          
        trace_fitnesses = [alignments['log_fitness']]
    
    elif method_fitness == "conformance_diagnostics_token_based_replay":
        alignments = pm4py.conformance_diagnostics_token_based_replay(log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [a['trace_fitness'] for a in alignments]
        
    elif method_fitness == "fitness_token_based_replay":
        alignments = pm4py.fitness_token_based_replay(log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [alignments['log_fitness']]
        
        
    trace_ids = list(log[log_data.case_name_key].unique())
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness


# === Helper functions ===


def encode(crop_trace: pd.DataFrame, log_data: LogData,  maxlen: int, char_indices: Dict[str, int],
                      char_indices_group: Dict[str, int], resource: bool) -> np.ndarray:
    """
    encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    if resource:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        sentence_group = ''.join(crop_trace[log_data.res_name_key].tolist())
        chars_group = list(char_indices_group.keys())
        if shared.One_hot_encoding:
            num_features = len(chars) + len(chars_group)
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if char in chars:
                    x[0, t + leftpad, char_indices[char] - 1] = 1
                if t < len(sentence_group):
                    if sentence_group[t] in chars_group:
                        x[0, t + leftpad, len(char_indices) + char_indices_group[sentence_group[t]] - 1] = 1
        elif shared.use_modulator:
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
            if shared.combined_Act_res:
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
        if shared.One_hot_encoding:
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

def repetitions(seq: str):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(seq):
        yield match.group(1), len(match.group(0)) / len(match.group(1))


def reduce_loop_probability(prefix_seq):
    list_of_rep = list(repetitions(prefix_seq))
    if list_of_rep:
        str_rep = list_of_rep[-1][0]
        if prefix_seq.endswith(str_rep):
            return np.math.exp(list_of_rep[-1][-1]), list_of_rep[-1][0][0]
        else:
            return 1, list_of_rep[-1][0][0]
    return 1, " "


def apply_reduction_probability(act_seq, res_seq, pred_act, pred_res, target_act_to_ind, target_res_to_ind, resource=False):
    stop_symbol_probability_amplifier_current, start_of_the_cycle_symbol=reduce_loop_probability(act_seq)
    if start_of_the_cycle_symbol in target_act_to_ind:
        place_of_starting_symbol = target_act_to_ind[start_of_the_cycle_symbol]
        pred_act[place_of_starting_symbol-1] = pred_act[place_of_starting_symbol-1] / stop_symbol_probability_amplifier_current
    stop_symbol_probability_amplifier_current_res, start_of_the_cycle_symbol_res = reduce_loop_probability(res_seq)
    if resource:
        if start_of_the_cycle_symbol_res in target_res_to_ind:
            place_of_starting_symbol_res = target_res_to_ind[start_of_the_cycle_symbol_res]
            pred_res[place_of_starting_symbol_res-1] = pred_res[
                                                     place_of_starting_symbol_res-1] / stop_symbol_probability_amplifier_current_res
    return pred_act, pred_res

def get_beam_size(self, NodePrediction, current_prediction_premis, bk_model,weight, prefix_trace,
                  prediction, res_prediction, y_char, fitness,
                  target_ind_to_act, target_act_to_ind, target_ind_to_res, target_res_to_ind,
                  log_data, resource, beam_size):
    record = []
    act_prefix = prefix_trace.cropped_line
    res_prefix = prefix_trace.cropped_line_group if resource else None

    print(f'Beam size: {beam_size}, act_prefix: {act_prefix}',f'res_prefix: {res_prefix}' if resource else '')
    prefix_trace = prefix_trace.cropped_trace if isinstance(prefix_trace, NodePrediction) else prefix_trace
    if shared.useProb_reduction:
        prediction, res_prediction =  apply_reduction_probability(act_prefix, res_prefix, prediction, res_prediction, target_act_to_ind, target_res_to_ind,
                               resource)
    if resource:
        # create probability matrix
        prob_matrix = np.log(prediction) + np.log(res_prediction[:, np.newaxis])
        if shared.declare_BK and not shared.BK_end:
            for res_pred_idx, act_pred_idx in np.ndindex(prob_matrix.shape):
                temp_prediction = target_ind_to_act[act_pred_idx + 1]
                temp_res_prediction = target_ind_to_res[res_pred_idx + 1]
                BK_res = compliance_checking(log_data, temp_prediction, temp_res_prediction, bk_model, prefix_trace,resource)
                prob_matrix[res_pred_idx][act_pred_idx] = (prob_matrix[res_pred_idx][act_pred_idx] * 0.5 * (1-weight)) + (BK_res * weight)
        sorted_prob_matrix = np.argsort(prob_matrix, axis=None)[::-1]
    else:
        if shared.declare_BK:
            BK_res = np.zeros(len(prediction), dtype=np.float32)
            for act_ped_index,_ in enumerate(prediction):
                temp_prediction = target_ind_to_act[act_ped_index + 1]
                BK_res [act_ped_index] = compliance_checking(log_data, temp_prediction, None, bk_model, prefix_trace,
                                             resource)
            prediction = [(np.log(a) * (1-weight)) + (b * weight) for a, b in zip(prediction, BK_res)]
        elif shared.prob_declare_BK:
            prefix_act = [log_data.act_enc_mapping[index] for index in act_prefix]
            results = bk_model.processPrefix(prefix_act, shared.aggregationMethod)
            BK_res = np.zeros(len(prediction), dtype=np.float32)
            target_acts = [log_data.act_enc_mapping[target_ind_to_act[indice]] if target_ind_to_act[
                                                                                      indice] != '!'  # list of activities
                           else 'False'  # End symbol
                           for indice in range(1, len(prediction) + 1)
                           ]
            for event, score in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
                if event is False:  # End symbol
                    BK_res[target_acts.index('False')] = round(score, 3)
                elif event is not True:  # Activities present in the declare model
                    BK_res[target_acts.index(event)] = round(score, 3)
                else:  # if event is True: Activities not present in the declare model
                    act_indices = [i for i, act in enumerate(target_acts) if
                                   act not in results.keys() and act != 'False']
                    for idx in act_indices:
                        BK_res[idx] = round(score, 3)
            prediction = [(np.log(a) * (1 - weight)) + (b * weight) for a, b in zip(prediction, BK_res)]
        else:
            prediction = prediction #np.log(prediction)

    for j in range(beam_size):
        prefix_trace = prefix_trace.cropped_trace if isinstance(prefix_trace, NodePrediction) else prefix_trace

        print(f'Iteration: {j}')
        if resource:
            res_pred_idx, act_pred_idx = np.unravel_index(sorted_prob_matrix[j], prob_matrix.shape)
            temp_prediction = target_ind_to_act[act_pred_idx + 1]
            temp_res_prediction = target_ind_to_res[res_pred_idx + 1]
            probability_this = prob_matrix[res_pred_idx][act_pred_idx]
        else:
            pred_idx = np.argsort(prediction)[len(prediction) - j - 1]
            temp_prediction = target_ind_to_act[pred_idx + 1]
            temp_res_prediction = None
            probability_this = np.sort(prediction)[len(prediction) - 1 - j]

        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_prediction
        if resource: predicted_row.loc[:, log_data.res_name_key] = temp_res_prediction
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)

        probability_of = (current_prediction_premis.probability_of + probability_this)

        print(f'Temp prediction: {temp_prediction}, Temp res prediction: {temp_res_prediction}, Probability:{probability_of}')
        temp = NodePrediction(temp_cropped_trace_next,probability_of)
        self.put(temp)
        trace_org = [log_data.act_enc_mapping[i] if i != "!" else "" for i in temp_cropped_trace_next[log_data.act_name_key].tolist()]
        if len(fitness) > 0:
            fitness_sorted = np.array(fitness)[np.argsort(prediction)]
            fitness_this = fitness_sorted[len(fitness_sorted) - 1 - j]
            y_char_sorted = np.array(y_char)[np.argsort(prediction)]
            y_char_this = y_char_sorted[len(y_char_sorted) - 1 - j]

            record.append(str(
                "trace_org = " + '>>'.join(trace_org) +
                "// previous = " + str(round(current_prediction_premis.probability_of, 3)) +
                "// current = " + str(round(current_prediction_premis.probability_of + probability_this, 3)) +
                "// rnn = " + str(round(y_char_this, 3)) +
                "// fitness = " + str(round(fitness_this, 3))) +
                            "&"
                            )
    return self, record

def compliance_checking(log_data, temp_prediction, temp_res_prediction, bk_model, prefix_trace, resource=False):

    BK_result = 0
    completed = False
    if temp_prediction == "!" or (resource and temp_res_prediction == "!"):
        completed = True
        temp_prediction = "!"
        if resource:
            temp_res_prediction = "!"
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
                                            consider_vacuity=True, completed=completed)
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

