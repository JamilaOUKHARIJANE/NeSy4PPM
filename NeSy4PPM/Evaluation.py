from pathlib import Path

import numpy as np
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from pm4py.objects.log.util import dataframe_utils
import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as petrinet_factory
from pm4py.objects.conversion.log import converter as log_converter
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import BK_type
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

from NeSy4PPM.Prediction.Checkers import TraceDeclareAnalyzer


def evaluate_all(output_folder, filename, metrics, log_data:LogData=None, resource:bool=False,
                 bk_model=None,fitness_method="fitness_token_based_replay",weight=None,prefix_len=None, traces_ids = [],
                 rbc_min=None, rbc_max=None):
    file_path = os.path.join(output_folder, filename)
    if not os.path.exists(file_path):
        results={}
        results["Time"] = {"Average time": 0, "Standard deviation time": 0}
        results["EBC"] = 0
        results['Damerau-Levenshtien similarity'] = {"Activities": 0, "Resources":  None}
        if 'Compliance' in metrics: results["Compliance"] = 0
        if "Fitness" in metrics: results["Fitness"] = 0
        if "SDFA Conformance" in metrics: results["SDFA Conformance"] = {
                "feasibility_rate": 0.0,
                "termination_rate": 0.0,
            }
        return results
    df_results = pd.read_csv(os.path.join(output_folder, filename), delimiter=',')
    if prefix_len is not None:
        df_results = df_results[df_results['Prefix length'] == prefix_len]
    if weight is not None:
        df_results = df_results[df_results["Weight"]== weight]
    if traces_ids :
        df_results['Case ID'] = df_results['Case ID'].astype(str).apply(
            lambda x: x if x.endswith('_test') else x + '_test')
        df_results = df_results[df_results['Case ID'].isin(traces_ids)]
    if "Fitness" in metrics or 'Compliance' in metrics:
        df_results['act'] = np.where(df_results['Predicted Acts'].notna() & (df_results['Predicted Acts'].str.strip() != ''),
                                     df_results['Trace Prefix Act'] + '>>'+df_results['Predicted Acts'],df_results['Trace Prefix Act'])
        if resource:
            df_results['res'] = np.where(df_results['Predicted Resources'].notna() & (df_results['Predicted Resources'].str.strip() != ''),
                                     df_results['Trace Prefix Res']+ '>>'+df_results['Predicted Resources'],df_results['Trace Prefix Res'])
        selected_columns = df_results[['Case ID','Prefix length','act', 'res']].copy() if resource else df_results[['Case ID','Prefix length','act']].copy()
        selected_columns['concept:name'] = selected_columns['act'].str.split('>>')
        selected_columns["time:timestamp"] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s')
        selected_columns['case:concept:name'] = selected_columns["Case ID"] + '_' + selected_columns['Prefix length'].astype(str)+ '_'+ selected_columns.index.astype(str)
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
        if metric == "SDFA Conformance":
            def split_activities(value):
                if pd.isna(value) or str(value).strip() == "":
                    return []
                return [activity.strip() for activity in str(value).split(">>") if activity.strip()]

            predictions = [
                (split_activities(row["Trace Prefix Act"]), split_activities(row["Predicted Acts"]))
                for _, row in df_results.iterrows()
            ]
            compliance_metrics = bk_model.compute_compliance_metrics(predictions)
            results[metric] = {
                key: round(value, 3)
                for key, value in compliance_metrics.items()
            }

        if metric == 'EBC' :
            EBC_values = pd.to_numeric(df_results["EBC"], errors="coerce").dropna()
            if EBC_values.empty or rbc_min == rbc_max:
                results[metric] = 0.0
            else:
                results[metric] = round(((EBC_values - rbc_min) / (rbc_max - rbc_min)).clip(0, 1).mean(), 3)

        if metric == 'Damerau-Levenshtien similarity':
            results[metric]= {"Activities": round(df_results['Damerau-Levenshtein Acts'].mean(),3),
                              "Resources": round(df_results['Damerau-Levenshtein Resources'].mean(),3) if resource else None}
        if metric == 'Jaccard similarity':
            results[metric] = {"Activities": round(df_results['Jaccard Acts'].mean(),3),
                               "Resources": round(df_results['Jaccard Resources'].mean(),3) if resource else None}
        if metric == "Compliance":
            log1["lifecycle:transition"] = "complete"
            log1 = dataframe_utils.convert_timestamp_columns_in_df(log1)
            event_log = log_converter.apply(log1)
            compliance = log_conformance(event_log, bk_model)
            results[metric] = compliance
        if metric == "Fitness":
            fintness = get_fitness(log1, bk_model,fitness_method)
            results[metric] = fintness
    return results


def log_conformance(log, bk_model):
    d_log = D4PyEventLog()
    d_log.log = log
    d_log.log_length = len(d_log.log)
    d_log.timestamp_key = 'time:timestamp'
    d_log.activity_key = 'concept:name'
    basic_checker = TraceDeclareAnalyzer(log=d_log, declare_model=bk_model,
                                         consider_vacuity=False, completed=True)
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
    gviz = petrinet_factory.apply(net, initial_marking, final_marking)
    petrinet_factory.view(gviz)
    return {"net": net, "initial_marking": initial_marking, "final_marking": final_marking,"type":BK_type.Procedural}

def discover_bpmn(log_path):
    event_log = xes_importer.apply(str(log_path))
    bpmn_graph = pm4py.discover_bpmn_inductive(event_log)
    gviz = bpmn_visualizer.apply(bpmn_graph)
    bpmn_visualizer.view(gviz)
    return bpmn_graph
