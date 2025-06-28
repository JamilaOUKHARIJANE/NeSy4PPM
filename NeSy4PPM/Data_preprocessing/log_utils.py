import pandas as pd
import pm4py
from pathlib import Path


class LogData:
    log: pd.DataFrame
    log_name: str
    train_log_name: str
    test_log_name: str
    max_len: int
    training_trace_ids = [str]
    evaluation_trace_ids = [str]

    # Gathered from encoding
    act_enc_mapping: {str, str}
    res_enc_mapping: {str, str}

    # Gathered from manual log analysis
    case_name_key: str
    act_name_key: str
    res_name_key: str
    timestamp_key: str
    timestamp_key2: str
    timestamp_key3: str
    timestamp_key4: str
    label_name_key: str
    label_pos_val: str
    label_neg_val: str
    compliance_th: float
    evaluation_th: float
    evaluation_prefix_start: int
    evaluation_prefix_end: int

    def __init__(self, log_path:Path,log_name=None,train_ratio=0.8, train_log=None, test_log=None, case_name_key = 'case:concept:name',act_name_key = 'concept:name'
                 ,res_name_key = 'org:resource',timestamp_key = 'time:timestamp'):
        self.case_name_key = case_name_key
        self.act_name_key = act_name_key
        self.res_name_key = res_name_key
        self.timestamp_key = timestamp_key
        self.test_log_name = test_log
        self.train_log_name= train_log
        self.log_name = Path(log_name).stem if log_name  else None
        if self.log_name is not None and self.train_log_name is None and self.test_log_name is None:
            self.log= self.read_log(log_path, log_name)
            trace_ids = self.log[self.case_name_key].unique().tolist()
            # Simple Train/Test Split based on shared variables
            sorting_cols = [self.timestamp_key, self.act_name_key]
            if self.res_name_key in self.log.columns:
                sorting_cols.append(self.res_name_key)
            self.log = self.log.sort_values(sorting_cols, ascending=True, kind='mergesort')
            grouped = self.log.groupby(self.case_name_key)
            start_timestamps = grouped[self.timestamp_key].min().reset_index()
            start_timestamps = start_timestamps.sort_values(self.timestamp_key, ascending=True, kind='mergesort')
            train_ids = list(start_timestamps[self.case_name_key])[:int(train_ratio * len(start_timestamps))]
            test_ids = [trace for trace in trace_ids if trace not in train_ids]
            # Outputs
            self.training_trace_ids = train_ids
            self.evaluation_trace_ids = test_ids
        elif self.train_log_name is not None and self.test_log_name is not None:
            train_log = self.read_log(log_path, self.train_log_name)
            test_log = self.read_log(log_path, self.test_log_name)
            self.log_name = Path(self.train_log_name).stem
            self.log = pd.concat([train_log, test_log], axis=0, ignore_index=True)
            self.training_trace_ids = train_log[self.case_name_key].unique().tolist()
            self.evaluation_trace_ids = test_log[self.case_name_key].unique().tolist()
        else:
            raise ValueError("An event log or a train_log with a test_log are required")
        self.encode_log(self.res_name_key in self.log.columns)
        trace_sizes = list(self.log.value_counts(subset=[self.case_name_key], sort=False))
        self.max_len = max(trace_sizes)

    def encode_log(self, resource: bool, ascii_offset = 161):
        act_set = list(self.log[self.act_name_key].unique())
        self.act_enc_mapping = dict((chr(idx+ascii_offset), elem) for idx, elem in enumerate(act_set))
        self.log.replace(to_replace={self.act_name_key: {v: k for k, v in self.act_enc_mapping.items()}}, inplace=True)
        if resource:
            res_set = list(self.log[self.res_name_key].unique())
            self.res_enc_mapping = dict((chr(idx+ascii_offset), elem) for idx, elem in enumerate(res_set))
            self.log.replace(to_replace={self.res_name_key: {v: k for k, v in self.res_enc_mapping.items()}}, inplace=True)

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
            log = pd.read_csv(log_path)
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





