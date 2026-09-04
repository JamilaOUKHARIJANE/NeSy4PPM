import copy
from collections import Counter
from statistics import median
import pandas as pd
import pm4py
from pathlib import Path
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.utils import get_properties

class LogData:
    log: pd.DataFrame
    log_name: str
    max_len: int
    median: int
    training_trace_ids = [str]
    feedback_trace_ids = [str]
    evaluation_trace_ids = [str]

    # Gathered from encoding
    act_enc_mapping: {str, str}
    res_enc_mapping: {str, str}
    target_int_to_act: {int, str}
    target_int_to_res: {int, str}
    act_to_int: {str, int}
    res_to_int: {str, int}

    # Gathered from manual log analysis
    case_name_key: str
    act_name_key: str
    res_name_key: str
    timestamp_key: str
    evaluation_prefix_start: int
    evaluation_prefix_end: int

    def __init__(self, log_path:Path,log_name=None,train_ratio=0.8,feedback_ratio=0, train_log=None, feedback_log=None,test_log=None, case_name_key = 'case:concept:name',act_name_key = 'concept:name'
                 ,res_name_key = 'org:resource',timestamp_key = 'time:timestamp', min_support=0.0):
        self.case_name_key = case_name_key
        self.act_name_key = act_name_key
        self.res_name_key = res_name_key
        self.timestamp_key = timestamp_key
        self.log_name = Path(log_name).stem if log_name else None

        # Simple Train/feedback/Test Split  (if you want feedback set use a feedback ratio different to 0)
        if self.log_name is not None and train_log is None and test_log is None:
            self.log= self.read_log(log_path, log_name)
            self.log = self.log.sort_values(by=[self.case_name_key, self.timestamp_key, self.act_name_key])
            self.remove_variant_outliers(min_support)
            grouped = self.log.groupby(self.case_name_key)
            start_timestamps = grouped[self.timestamp_key].min().reset_index()
            start_timestamps = start_timestamps.sort_values(self.timestamp_key, ascending=True, kind='mergesort')
            case_ids = start_timestamps[self.case_name_key].tolist()
            self.training_trace_ids = case_ids[:int(train_ratio * len(case_ids))]
            self.feedback_trace_ids = case_ids[int(train_ratio * len(case_ids)):int((train_ratio+feedback_ratio) * len(case_ids))]
            self.evaluation_trace_ids = case_ids[int((train_ratio+feedback_ratio) * len(case_ids)):]
        # Read Train, feedback and Test sets (feedback is optional) from .csv or .xes or .xes.gz file or a Dataframe
        elif train_log is not None and test_log is not None:
            if not isinstance(train_log, pd.DataFrame): train_log = self.read_log(log_path, train_log)
            if feedback_log is not None and not isinstance(feedback_log, pd.DataFrame):
                feedback_log = self.read_log(log_path, feedback_log)
                feedback_log[self.case_name_key] = feedback_log[self.case_name_key].astype(str).apply(lambda x: x if x.endswith('_feedback') else x + '_feedback')
            if not isinstance(test_log, pd.DataFrame): test_log = self.read_log(log_path, test_log)
            test_log[self.case_name_key] = test_log[self.case_name_key].astype(str).apply(lambda x: x if x.endswith('_test') else x + '_test')
            logs_to_concat = [train_log, test_log] if feedback_log is None else [train_log, feedback_log, test_log]
            self.log = pd.concat(logs_to_concat, axis=0, ignore_index=True)
            cases_ids= self.log[self.case_name_key].tolist()
            self.training_trace_ids = train_log[train_log[self.case_name_key].isin(cases_ids)][self.case_name_key].unique().tolist()
            self.feedback_trace_ids = feedback_log[feedback_log[self.case_name_key].isin(cases_ids)][self.case_name_key].unique().tolist() if feedback_log is not None else []
            self.evaluation_trace_ids = test_log[test_log[self.case_name_key].isin(cases_ids)][self.case_name_key].unique().tolist()
        else:
            raise ValueError("An event log or a train_log with a test_log is required")
        self.encode_log(self.res_name_key in self.log.columns)
        trace_sizes = list(self.log.value_counts(subset=[self.case_name_key], sort=False))
        self.max_len = max(trace_sizes)
        self.median = median(trace_sizes)

    def remove_variant_outliers(self, min_support):
        def get_variant_supports(event_log) -> Counter:
            return Counter([",".join(event["concept:name"] for event in trace) for trace in event_log])

        parameters = get_properties(self.log, case_id_key=self.case_name_key,
                                    activity_key=self.act_name_key,
                                    timestamp_key=self.timestamp_key)
        log = log_converter.apply(self.log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

        variant_supports = {variant for variant, count in get_variant_supports(log).items()}
        keep_varaints = variant_supports.copy()
        for v in variant_supports:
            if (get_variant_supports(log)[v] < min_support * (len(log))):  # filter the log to keep variants that have at least more than 1% log traces
                keep_varaints.remove(v)
        filtered_traces = [trace for trace in log if ",".join(event["concept:name"] for event in trace) in keep_varaints]
        case_ids = list({trace.attributes["concept:name"] for trace in filtered_traces})
        self.log = self.log[self.log[self.case_name_key].isin(case_ids)]


    def encode_log(self, resource: bool, ascii_offset = 161):
        act_set = list(self.log[self.act_name_key].unique())
        self.act_enc_mapping = dict((chr(idx + ascii_offset), elem) for idx, elem in enumerate(act_set))
        self.log.replace(to_replace={self.act_name_key: {v: k for k, v in self.act_enc_mapping.items()}}, inplace=True)
        act_chars = self.log[self.act_name_key].unique().tolist()
        act_chars.sort()
        self.act_to_int = dict((c, i + 1) for i, c in enumerate(act_chars))
        target_act_chars = copy.copy(act_chars)
        target_act_chars.append('!')
        self.target_int_to_act = dict((i + 1, c) for i, c in enumerate(target_act_chars))
        self.res_to_int = None
        if resource:
            res_set = list(self.log[self.res_name_key].unique())
            self.res_enc_mapping = dict((chr(idx+ascii_offset), elem) for idx, elem in enumerate(res_set))
            self.log.replace(to_replace={self.res_name_key: {v: k for k, v in self.res_enc_mapping.items()}}, inplace=True)
            res_chars = list(self.log[self.res_name_key].unique())
            res_chars.sort()
            target_res_chars = copy.copy(res_chars)
            self.res_to_int = dict((c, i + 1) for i, c in enumerate(res_chars))
            target_res_chars.append('!')
            self.target_int_to_res = dict((i + 1, c) for i, c in enumerate(target_res_chars))

    def read_log(self, log_path, log_name):
        if log_name.endswith('.xes') or log_name.endswith('.xes.gz'):
            log_path = log_path / log_name
            log = pm4py.read_xes(str(log_path))
            cols = [self.case_name_key, self.act_name_key, self.timestamp_key]
            if self.res_name_key in log.columns:
                cols.append(self.res_name_key)
            log=log[cols]
            log[self.timestamp_key] = pd.to_datetime(log[self.timestamp_key])
        elif log_name.endswith('.csv'):
            log = pd.read_csv(log_path/ log_name)
            log.columns = log.columns.str.strip()
            cols = [self.case_name_key, self.act_name_key, self.timestamp_key]
            if self.res_name_key in log.columns:
                cols.append(self.res_name_key)
            log = log[cols]
            log[self.case_name_key] = log[self.case_name_key].astype(str)
            log[self.timestamp_key] = pd.to_datetime(log[self.timestamp_key])
        else:
            raise RuntimeError(f"Extension of {log_name} must be in ['.xes', '.xes.gz', '.csv'].")
        return log

    def prepare_encoded_data(self, resource: bool):
        """
        Get all possible symbols for activities and resources and annotate them with integers.
        """
        act_chars = self.log[self.act_name_key].unique().tolist()
        act_chars.sort()
        target_act_chars = copy.copy(act_chars)
        target_act_chars.append('!')

        act_to_int = dict((c, i + 1) for i, c in enumerate(act_chars))
        target_act_to_int = dict((c, i + 1) for i, c in enumerate(target_act_chars))
        target_int_to_act = dict((i + 1, c) for i, c in enumerate(target_act_chars))

        if resource:
            res_chars = list(self.log[self.res_name_key].unique())
            res_chars.sort()
            target_res_chars = copy.copy(res_chars)
            target_res_chars.append('!')
            res_to_int = dict((c, i + 1) for i, c in enumerate(res_chars))
            target_res_to_int = dict((c, i + 1) for i, c in enumerate(target_res_chars))
            target_int_to_res = dict((i + 1, c) for i, c in enumerate(target_res_chars))
        else:
            res_chars = None
            res_to_int = None
            target_res_to_int = None
            target_int_to_res = None
        return act_chars, res_chars, act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res





