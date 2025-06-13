from enum import Enum
from pathlib import Path

import numpy as np
import pm4py

from src.commons import log_utils, shared_variables as shared
import csv
import os
import pandas as pd
from statistics import mean

from src.commons.shared_variables import BK_end


class LogName(Enum):
    SEPSIS = 'Sepsis_cases'
    HELPDESK = 'helpdesk'
    BPIC11 = "BPI2011"
    BPIC12 = "BPI2012"
    BPIC13 = "BPI2013"
    BPIC13_In = "BPI2013_In"
    BPIC13_OP = "BPI2013_OP"
    BPIC13_CP = "BPI2013_CP"
    BPIC15_1 = "BPIC15_1"
    BPIC15_2 = "BPIC15_2"
    BPIC15_3 = "BPIC15_3"
    BPIC15_4 = "BPIC15_4"
    BPIC15_5 = "BPI2015_5"
    BPIC17 = "BPI2017"
    BPIC18 = "BPI2018"
    BPIC19 = "BPIC19"
    BPIC20_1 = "PrepaidTravelCost"
    BPIC20_2 = "PermitLog"
    BPIC20_3 = "RequestForPayment"
    BPIC20_4 = "DomesticDeclarations"
    BPIC20_5 = "InternationalDeclarations"

def set_log_ths(log_path):
    test_log = pm4py.read_xes(str(log_path).replace(".xes", f"_filtred.xes"))
    #trace_ids = None
    trace_ids = test_log['case:concept:name'].drop_duplicates().tolist()
    log_name = log_path.stem
    if log_name == LogName.HELPDESK.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC11.value:
        median = 92
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC12.value:
        median = 32
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC13.value:
        median = 4
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC17.value:
        median = 54
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC13_In.value:
        median = 6
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC13_CP.value:
        median = 3
        evaluation_prefix_start = median // 2
    elif log_name == LogName.SEPSIS.value:
        median = 13
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_1.value:
        median = 44
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_2.value:
        median = 54
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_3.value:
        median = 42
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_4.value:
        median = 44
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC15_5.value:
        median = 50
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC20_1.value:
        median = 8
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC20_2.value:
        median = 11
        evaluation_prefix_start = median // 2 - 2
    elif log_name == LogName.BPIC20_3.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC20_4.value:
        median = 5
        evaluation_prefix_start = median // 2 - 1
    elif log_name == LogName.BPIC20_5.value:
        median = 10
        evaluation_prefix_start = median // 2 - 2
    else:
        raise RuntimeError(f"No settings defined for log: {log_name.value}.")
    return evaluation_prefix_start, trace_ids

save_log=True
def aggregate_results(log_path, alg,models_folder,beam_size=3, resource=False,timestamp=False,outcome=False,probability_reduction=False, BK=False, BK_end=False, weight=0.0):
    average_act=[]
    average_res =[]
    average_length = []
    average_length_truth = []
    average_length_res = []
    all_folds_data = []
    log_name = log_path.stem + "_filtred"
    if save_log:
        log_data = log_utils.LogData(log_path, log_name.__add__('.xes'))
        log_data.encode_log(resource, False, False)
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        folder_path = shared.output_folder / models_folder / str(fold) / 'results_prody' / eval_algorithm
        #if  alg=='baseline': folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        #print(f"fold {fold} - {eval_algorithm}")
        evaluation_prefix_start, test_trace_ids = set_log_ths(log_path)
        if alg == "beamsearch":
            filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{BK_end*"_BK_END"}{f"_{shared.BK_type}" * BK}.csv'
        else:
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0,0,0
        df_results = pd.read_csv(file_path, delimiter=',')
        df_results = df_results[df_results['Weight'] == weight]
        df_results = df_results[df_results['Case ID'].isin(test_trace_ids)] if alg == "beamsearch" else df_results[df_results['Case Id'].isin(test_trace_ids)]
        if resource:
            if log_name.startswith('helpdesk'):
                df_results = df_results[df_results['Prefix length'] < 5]
            average_act.append(df_results['Damerau-Levenshtein Acts'].mean())
            average_res.append(df_results['Damerau-Levenshtein Resources'].mean())
            average_length.append(df_results['Predicted Acts'].str.len().mean())
            average_length_res.append(df_results['Predicted Resources'].str.len().mean())
            average_length_truth.append(df_results['Ground truth'].str.len().mean())
            df_results['act'] = np.where(
                df_results['Predicted Acts'].notna() & (df_results['Predicted Acts'].str.strip() != ''),
                df_results['Trace Prefix Act'] + df_results['Predicted Acts'],
                df_results['Trace Prefix Act'])
            df_results['res'] = np.where(
                df_results['Predicted Resources'].notna() & (df_results['Predicted Resources'].str.strip() != ''),
                df_results['Trace Prefix Res'] + df_results['Predicted Resources'],
                df_results['Trace Prefix Res'])
            selected_columns = df_results[
                ['Case Id' if 'Case Id' in df_results.columns else 'Case ID','Prefix length','act', 'res' ]].copy()
        else:
            average_act.append(df_results['Damerau-Levenshtein'].mean())
            average_length.append(df_results['Predicted'].str.len().mean())
            average_length_truth.append(df_results['Ground truth'].str.len().mean())
            df_results['act'] = np.where(
                df_results['Predicted'].notna() & (df_results['Predicted'].str.strip() != ''),
                df_results['Trace Prefix Act'] + df_results['Predicted'],
                df_results['Trace Prefix Act'])
            selected_columns = df_results[
                ['Case Id' if 'Case Id' in df_results.columns else 'Case ID','Prefix length','act']].copy()
        if save_log:
            if 'Case Id' in selected_columns.columns:
                selected_columns = selected_columns.rename(columns={'Case Id': 'Case ID'})
            selected_columns['concept:name'] = selected_columns['act'].apply(
                    lambda x: (",".join(log_data.act_enc_mapping[char] for char in x) if isinstance(x, str) else x))
            selected_columns["time:timestamp"] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s')
            selected_columns['case:concept:name'] = selected_columns["Case ID"] + f'_{fold}_' + selected_columns[
                    'Prefix length'].astype(
                    str)
            if resource:
                selected_columns['org:resource'] = selected_columns['res'].apply(
                        lambda x: (",".join(log_data.res_enc_mapping[char] for char in x) if isinstance(x, str) else x))
                # selected_columns = selected_columns[selected_columns['act'].notna() & (selected_columns['act'].str.strip() != '')]
            selected_columns['concept:name'] = selected_columns['concept:name'].str.split(',')
            if resource:
                selected_columns['org:resource'] = selected_columns['org:resource'].str.split(',')
                log1 = selected_columns.explode(['concept:name', 'org:resource'], ignore_index=True)
                log1['concept:name'] = log1['concept:name'].str.strip()
                log1['org:resource'] = log1['org:resource'].str.strip()
                log1 = log1[['case:concept:name', 'concept:name', 'org:resource', 'time:timestamp']]
            else:
                log1 = selected_columns.explode(['concept:name'], ignore_index=True)
                log1['concept:name'] = log1['concept:name'].str.strip()
                log1 = log1[['case:concept:name', 'concept:name', 'time:timestamp']]
            all_folds_data.append(log1)
    if save_log:
        df_all_folds = pd.concat(all_folds_data, ignore_index=True)
        if not Path.exists(Path.cwd() /'predictedlog'/ models_folder):
            Path.mkdir(Path.cwd() /'predictedlog'/ models_folder, parents=True)
        save_path = Path.cwd() /'predictedlog'/ models_folder / f"{log_name}_{eval_algorithm}_{weight}_{BK*(shared.BK_type or '')}{BK_end * 'BK_end'}.csv"
        df_all_folds.to_csv(save_path, index=False)
    if resource:
        print(f"{log_name}_{models_folder} - {eval_algorithm} -{BK * 'BK'}-{BK_end * 'BK_END'}", ":", round(mean(average_act), 3), ":",
              round(mean(average_res), 3), ":"
              , round(mean(average_length), 3), ":",         round(mean(average_length_truth), 3)
              )
        return round(mean(average_act), 3) , round(mean(average_res), 3), round(mean(average_length), 3) , round(mean(average_length_truth), 3)
    else:
        print(f"{log_name}_{models_folder} - {eval_algorithm}-{BK * 'BK'}-{BK_end * 'BK_END'}", ":", round(mean(average_act), 3), ":", round(mean(average_length), 3), ":",
              round(mean(average_length_truth), 3))
        return round(mean(average_act), 3), 0.0, round(mean(average_length), 3) , round(mean(average_length_truth), 3)


def getresults(log_list, algo, encoder, models_folder, beam_size=3 ,resource=False, BK=False, BK_end=False, weight=0.0):
    results = []
    for log in log_list:
        log_path = shared.log_folder / log
        average_act, average_res, _, _ = aggregate_results(log_path, algo, models_folder + encoder, beam_size=beam_size, resource=resource, BK=BK, BK_end= BK_end, weight=weight)
        results.append(average_act)
        if resource:
            results.append(average_res)
    return results

if __name__ == "__main__":
   resource = False
   encoders = ["_One_hot","_Simple_categorical"] if not resource else ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"] #
   beam_sizes = [0,3] #,5,10
   weights = [i / 10 for i in range(5, 10)] +[0.85, 0.95,1.0]
   log_list = ['helpdesk.xes', 'Sepsis_cases.xes']#shared.log_list

   with (open(os.path.join(shared.output_folder, f"aggregated_results.csv"), mode='w') as out_file):
        writer = csv.writer(out_file, delimiter=',')
        headers = ["Method", "", "Encoder"]
        sub_headers = ["", "", ""]
        for log in log_list:
            headers.extend([log, ""]) if resource else headers.extend([log])
            sub_headers.extend(["Activities", "Resources"])  if resource else sub_headers.extend(["Activities"])
        headers.extend(['weight'])
        writer.writerow(headers)
        writer.writerow(sub_headers)
        for models_folder in ["keras_trans"]:
            for beam_size in beam_sizes:
                if beam_size == 0:
                    algo = 'baseline'
                    for encoder in encoders:
                        results = getresults(log_list,algo, encoder, models_folder, resource=resource)
                        writer.writerow([algo, beam_size, encoder.removeprefix("_")]+[res for res in results]+[0.0])
                else:
                    algo = 'beamsearch'
                    for encoder in encoders: #BS
                        shared.BK_type = 'Petri_net'
                        results = getresults(log_list, algo,encoder,models_folder,beam_size=beam_size, resource=resource,BK=True, BK_end=False)
                        writer.writerow([algo,beam_size,encoder.removeprefix("_")]+[res for res in results]+[0.0])
