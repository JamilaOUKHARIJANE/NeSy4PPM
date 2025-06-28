import copy
import itertools
from pathlib import Path

import numpy as np

from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import Encodings, prepare_encoded_data


def extract_trace_prefixes(log_data: LogData, resource: bool =False):
    """
    Extract activity and resource sequences starting from a list of trace ids (i.e. trace_names).
    """
    act_seqs = []  # list of all the activity sequences
    res_seqs = []  # list of all the resource sequences

    traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.training_trace_ids)]
    for _, trace in traces.groupby(log_data.case_name_key):
        line = ''.join(map(str ,trace[log_data.act_name_key].tolist()))  # sequence of activities for one case
        act_seqs.append(line)

        if resource:
            line_group = ''.join(map(str ,trace[log_data.res_name_key].tolist()))  # sequence of groups for one case
            res_seqs.append(line_group)
    prefixes ={'acts': act_seqs, 'res': res_seqs}
    return prefixes

def encode_prefixes(log_data: LogData ,prefixes, encoder:Encodings=Encodings.Index_based,resource: bool =False) -> tuple:
    chars, chars_group, act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res \
        = prepare_encoded_data(log_data, resource)
    # Adding '!' to identify end of trace
    training_lines = [x + '!' for x in prefixes['acts']]

    if resource:
        training_lines_group = [x + '!' for x in prefixes['res']]
        target_chars_group = copy.copy(chars_group)
        target_chars_group.append('!')
        print(f'Total resources: {len(chars_group)} - Target resources: {len(target_chars_group)}')
        print('\t', [log_data.res_enc_mapping[i] for i in chars_group])
    else:
        target_chars_group = None
    maxlen = log_data.max_len

    # Next lines here to get all possible characters for events and annotate them with numbers
    target_chars = copy.copy(chars)
    target_chars.append('!')

    print(f'Total activities: {len(chars)} - Target activities: {len(target_chars)}')
    print('\t', [log_data.act_enc_mapping[i] for i in chars])

    step = 1
    softness = 0
    sentences = []
    sentences_group = []
    next_chars = []
    next_chars_group = []

    if resource:
        if len(training_lines) != len(training_lines_group):
            raise ValueError("Mismatch in length of training_lines and training_lines_group")
        for line, line_group in zip(training_lines, training_lines_group):
            if len(line) != len(line_group):
                raise ValueError("Mismatch in length of line and line_group")

            for i in range(1, len(line)):
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_group.append(line_group[0: i])

                next_chars.append(line[i])
                next_chars_group.append(line_group[i])

        print('Num. of Training sequences:', len(sentences))
        print('Encoding...')
        if encoder == Encodings.One_hot:
            num_features = len(chars) + len(chars_group) # + 1
            x = np.zeros((len(sentences) ,maxlen, num_features), dtype=np.float32)
        else:
            if encoder == Encodings.Shrinked_based:
                result_list = [x + y for x, y in itertools.product(chars, chars_group)]
                target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
                num_features = maxlen
                x = np.zeros((len(sentences), num_features), dtype=np.float32)
            elif encoder == Encodings.Index_based:
                num_features = maxlen * 2
                x = np.zeros((len(sentences), num_features), dtype=np.float32)
            elif encoder == Encodings.Multi_encoders:
                num_features = maxlen
                x_a = np.zeros((len(sentences), num_features), dtype=np.float32)
                x_g = np.zeros((len(sentences), num_features), dtype=np.float32)
                x = {
                    "x_act": x_a,
                    "x_group": x_g
                }
            else:
                raise ValueError('Unknown encoder:', encoder)

        print(f'Num. of features: {num_features}')
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = np.zeros((len(sentences), len(target_chars_group)), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            counter_act = 0
            counter_res = 1
            sentence_group = sentences_group[i]
            for t, char in enumerate(sentence):
                if encoder == Encodings.One_hot:
                    if char in chars:
                        x[i, t + leftpad, act_to_int[char] -1] = 1
                    if t < len(sentence_group) and sentence_group[t] in chars_group:
                        x[i, t + leftpad, len(chars) + res_to_int[sentence_group[t]] - 1] = 1
                elif encoder == Encodings.Multi_encoders:
                    x_a[i, t] = act_to_int[char]
                    x_g[i, t] = res_to_int[sentence_group[t]]
                elif encoder == Encodings.Shrinked_based:
                    x[i, t] = target_to_int[char + sentence_group[t]]
                else:
                    x[i, counter_act] = act_to_int[char]
                    x[i, counter_res] = res_to_int[sentence_group[t]]
                    counter_act += 2
                    counter_res += 2

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_act_to_int[c] - 1] = 1 - softness
                else:
                    y_a[i, target_act_to_int[c] - 1] = softness / (len(target_chars) - 1)
            for c in target_chars_group:
                if c == next_chars_group[i]:
                    y_g[i, target_res_to_int[c] - 1] = 1 - softness
                else:
                    y_g[i, target_res_to_int[c] - 1] = softness / (len(target_chars_group) - 1)
        return x, y_a, y_g
    else:
        for line in training_lines:
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                next_chars.append(line[i])

        print('Num. of Training sequences:', len(sentences))

        if encoder == Encodings.One_hot:
            num_features = len(chars) + 1
            x = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        elif encoder == Encodings.Index_based:
            num_features = maxlen
            x = np.zeros((len(sentences), num_features), dtype=np.float32)
        else:
            raise ValueError('Unsupported encoder for only one attribute:', encoder)

        print(f'Num. of features: {num_features}')
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = None
        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if encoder == Encodings.One_hot:
                    for c in chars:
                        if c == char:
                            x[i, t + leftpad, act_to_int[c] - 1] = 1
                else:
                    x[i, t] = act_to_int[char]
            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_act_to_int[c] - 1] = 1 - softness
                else:
                    y_a[i, target_act_to_int[c] - 1] = softness / (len(target_chars) - 1)
        return x, y_a, y_g


def extract_encode_prefixes(log_data: LogData, encoder: Encodings = Encodings.Index_based, resource: bool = False):
    prefixes = extract_trace_prefixes(log_data, resource)
    x, y_a, y_g = encode_prefixes(log_data, prefixes, encoder, resource)
    return x, y_a, y_g

def end_to_end_process(log_path:Path,log_name=None,train_ratio=0.8, train_log=None, test_log=None, case_name_key = 'case:concept:name',act_name_key = 'concept:name'
                 ,res_name_key = 'org:resource',timestamp_key = 'time:timestamp',encoder: Encodings = Encodings.Index_based, resource: bool = False):
    log_data = LogData(log_path=log_path,log_name=log_name,train_ratio=train_ratio, train_log=train_log, test_log=test_log,
                                 case_name_key = case_name_key,act_name_key = act_name_key,res_name_key = res_name_key,timestamp_key = timestamp_key)
    x, y_a, y_g = extract_encode_prefixes(log_data, encoder=encoder, resource=resource)
    return log_data,x,y_a,y_g
