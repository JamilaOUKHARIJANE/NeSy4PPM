import numpy as np
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from pm4py.objects.log.util import dataframe_utils
import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as vis_factory
from pm4py.objects.conversion.log import converter as log_converter
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import NN_model, Encodings


def evaluate_all(log_data:LogData, model_arch:NN_model, encoder:Encodings, output_folder, filename, metrics, resource:bool=False,declare_model=None,
                 petri_net_model=None, fitness_method=None):
    models_folder = model_arch.value + "_" + encoder.value
    eval_algorithm = "CF" + "R"*resource
    folder_path = output_folder / models_folder / 'results' / eval_algorithm
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    df_results = pd.read_csv(file_path, delimiter=',')
    if "Fitness" in metrics or 'Compliance' in metrics:
        df_results['act'] = np.where(df_results['Predicted Acts'].notna() & (df_results['Predicted Acts'].str.strip() != ''),
                                     df_results['Trace Prefix Act'] + '>>'+df_results['Predicted Acts'],df_results['Trace Prefix Act'])
        if resource:
            df_results['res'] = np.where(df_results['Predicted Resources'].notna() & (df_results['Predicted Resources'].str.strip() != ''),
                                     df_results['Trace Prefix Res']+ '>>'+df_results['Predicted Resources'],df_results['Trace Prefix Res'])
        selected_columns = df_results[['Case ID','Prefix length','act', 'res']].copy() if resource else df_results[['Case ID','Prefix length','act']].copy()
        selected_columns['concept:name'] = selected_columns['act'].str.split('>>')
        selected_columns["time:timestamp"] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s')
        selected_columns['case:concept:name'] = selected_columns["Case ID"] + '_' + selected_columns['Prefix length'].astype(str)
        if resource:
            selected_columns['org:resource'] = selected_columns['res'].str.split('>>')
            log1 = selected_columns.explode(['concept:name', 'org:resource'], ignore_index=True)
            log1['org:resource'] = log1['org:resource'].str.strip()
            log1['concept:name'] = log1['concept:name'].str.strip()
            log1 = log1[['case:concept:name', 'concept:name', 'org:resource', 'time:timestamp']]
        else:
            log1 = selected_columns.explode(['concept:name'], ignore_index=True)
            log1['concept:name'] = log1['concept:name'].str.strip()
            log1 = log1[['case:concept:name', 'concept:name', 'time:timestamp']]
    results ={}
    for metric in metrics:
        if metric == 'Time':
            average_time=round(df_results['Time'].mean(),3)
            std_time =round(df_results['Time'].std(),3)
            results[metric]= {"Average time": average_time, "Standard deviation time": std_time}
        if metric == 'Damerau-Levenshtien similarity':
            results[metric]= {"Activities": round(df_results['Damerau-Levenshtein Acts'].mean(),3), "Resources": round(df_results['Damerau-Levenshtein Resources'].mean(),3) if resource else None}
        if metric == 'Jaccard similarity':
            results[metric] = {"Activities": round(df_results['Jaccard Acts'].mean(),3),
                               "Resources": round(df_results['Jaccard Resources'].mean(),3) if resource else None}
        if metric == "Compliance":
            log1["lifecycle:transition"] = "complete"
            log1 = dataframe_utils.convert_timestamp_columns_in_df(log1)
            event_log = log_converter.apply(log1)
            compliance = log_conformance(event_log, declare_model["model"])
            results[metric] = compliance
        if metric == "Fitness":
            fintness = get_fitness(log1, petri_net_model,fitness_method)
            results[metric] = fintness
    return results


def log_conformance(log, bk_model):
    d_log = D4PyEventLog()
    d_log.log = log
    d_log.log_length = len(d_log.log)
    d_log.timestamp_key = 'time:timestamp'
    d_log.activity_key = 'concept:name'
    #d_log.parse_xes_log(str(log_path))
    basic_checker = MPDeclareAnalyzer(log=d_log, declare_model=bk_model, consider_vacuity=True)
    conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
    state = conf_check_res.get_metric(metric="state")
    total_traces = len(state)
    state['sat'] = (state == 1).all(axis=1)
    satisfied_traces = sum(1 for v in state['sat'] if v)
    compliance = (satisfied_traces / total_traces) if total_traces > 0 else 0.0
    return round(compliance,3)

def get_fitness(event_log,bk_model,method_fitness= "fitness_token_based_replay"):
    net = bk_model["net"]
    initial_marking = bk_model["initial_marking"]
    final_marking = bk_model["final_marking"]
    if method_fitness == "conformance_diagnostics_alignments":
        alignments = pm4py.conformance_diagnostics_alignments(event_log, net, initial_marking, final_marking)
        trace_fitnesses = [a['fitness'] for a in alignments]
    elif method_fitness == "fitness_alignments":
        alignments = pm4py.fitness_alignments(event_log, net, initial_marking, final_marking)
        trace_fitnesses = alignments['log_fitness']
    elif method_fitness == "conformance_diagnostics_token_based_replay":
        alignments = pm4py.conformance_diagnostics_token_based_replay(event_log, net, initial_marking, final_marking)
        trace_fitnesses = [a['trace_fitness'] for a in alignments]
    elif method_fitness == "fitness_token_based_replay":
        alignments = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking)
        trace_fitnesses = alignments['log_fitness']
    return trace_fitnesses

def discover_petri_net(log_path):
    event_log = xes_importer.apply(str(log_path))
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
    gviz = vis_factory.apply(net, initial_marking, final_marking)
    vis_factory.view(gviz)
    return {"net": net, "initial_marking": initial_marking, "final_marking": final_marking}

