import csv
import os
from pathlib import Path

import pandas as pd
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessModels import DeclareModel
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log.get import get_variants
from pm4py.utils import get_properties
from pm4py.visualization.petri_net import visualizer as vis_factory
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import NeSy4PPM.commons.shared_variables as shared

from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from NeSy4PPM.commons import log_utils
from NeSy4PPM.prediction.prepare_data import get_pn_fitness

case_name_key = 'case:concept:name'
act_name_key = 'concept:name'
timestamp_key = 'time:timestamp'
def read_log(log_path):
    if str(log_path).endswith('.csv'):
        log = pd.read_csv(log_path)
        log["lifecycle:transition"] = "complete"
        log = dataframe_utils.convert_timestamp_columns_in_df(log)
        event_log = log_converter.apply(log)
        log_path = shared.root_folder /'predictedlog' /str(log_path).replace('.csv','.xes.gz')
        xes_exporter.apply(event_log, str(log_path), parameters={"gzip": True})
    else:
        event_log = xes_importer.apply(str(log_path))
    return event_log

def petri_net_model(log_path, df=None, set=''):
    if df is not None:
        log = dataframe_utils.convert_timestamp_columns_in_df(df)
        event_log = log_converter.apply(log)
    else:
        event_log = read_log(log_path)
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
    gviz = vis_factory.apply(net, initial_marking, final_marking)
    vis_factory.save(gviz, str(shared.root_folder/'predictedlog'/log_path.stem) +f'_{set}.png')
    vis_factory.view(gviz)
    alignments = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                  activity_key=act_name_key,
                                                  case_id_key=case_name_key,
                                                  timestamp_key=timestamp_key)
    trace_fitnesses = [alignments['log_fitness']]
    print(trace_fitnesses)
    return net, initial_marking, final_marking


def declareModel(log_path, thresold):
    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.parse_xes_log(str(log_path))
    discovery = DeclareMiner(log=event_log, consider_vacuity=False,
                             min_support=thresold,
                             itemsets_support=0.2,
                             max_declare_cardinality=1)
    discovered_model: DeclareModel = discovery.run()
    log_name= log_path.stem
    discovered_model.to_file(shared.declare_folder/log_name.__add__(f'_{thresold}.decl'))

log_path = Path.cwd() / 'predictedlog'

with (open(os.path.join(log_path, f"fitness_results.csv"), mode='w') as out_file):
    writer = csv.writer(out_file, delimiter=',')
    headers = ["log", "Encoder", "SAP", "BS", "BK_end", "petri_net"]
    for log in ['Sepsis_cases_filtred.xes','helpdesk_filtred.xes']: #
        net, initial_marking, final_marking = petri_net_model(log_path/log)
        for encoder in ["_Simple_categorical","_One_hot"]:
            encoder = "keras_trans"+encoder
            encoder_results=[]
            weight = 0.9
            alg = 'beamsearch'
            event_log = read_log(
                str(log_path / encoder / f"{log.removesuffix('.xes')}_{alg}_cf_{weight}_Petri_net.csv"))
            alignments_bk = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                             activity_key=act_name_key,
                                                             case_id_key=case_name_key,
                                                             timestamp_key=timestamp_key)
            print(alignments_bk['log_fitness'])
            for alg in ['baseline', 'beamsearch']:
                if alg == 'baseline':
                    event_log = read_log(log_path/encoder/f'{log.removesuffix(".xes")}_{alg}_cf_0.0_.csv')
                    alignments = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                                  activity_key=act_name_key,
                                                                  case_id_key=case_name_key,
                                                                  timestamp_key=timestamp_key)
                    encoder_results.append(alignments['log_fitness'])
                else:
                    event_log = read_log(log_path / encoder / f'{log.removesuffix(".xes")}_{alg}_cf_0.0_.csv')
                    alignments_bs = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                                  activity_key=act_name_key,
                                                                  case_id_key=case_name_key,
                                                                  timestamp_key=timestamp_key)
                    encoder_results.append(alignments_bs['log_fitness'])
                    #BK_end
                    event_log = read_log(log_path / encoder / f'{log.removesuffix(".xes")}_{alg}_cf_0.0_BK_end.csv')
                    alignments_bkend = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                                  activity_key=act_name_key,
                                                                  case_id_key=case_name_key,
                                                                  timestamp_key=timestamp_key)
                    encoder_results.append(alignments_bkend['log_fitness'])
                    #BS + BK
                    #["_One_hot", "_Simple_categorical", "_Combined_Act_res", "_Multi_Enc"]:
                    weight = 1.0  # 0.8 if encoder=='_Multi_Enc' else 0.6 if encoder=='_Simple_categorical' else 0.7
                    event_log = read_log(
                        str(log_path / encoder / f"{log.removesuffix('.xes')}_{alg}_cf_{weight}_Petri_net.csv"))

                    weight = 0.95  # 0.8 if encoder=='_Multi_Enc' else 0.6 if encoder=='_Simple_categorical' else 0.7
                    event_log = read_log(
                        str(log_path / encoder / f"{log.removesuffix('.xes')}_{alg}_cf_{weight}_Petri_net.csv"))
                    alignments_bk = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                                     activity_key=act_name_key,
                                                                     case_id_key=case_name_key,
                                                                     timestamp_key=timestamp_key)
                    encoder_results.append(alignments_bk['log_fitness'])
                    weight = 0.5 if log.startswith('helpdesk') else 0.85  # 0.8 if encoder=='_Multi_Enc' else 0.6 if encoder=='_Simple_categorical' else 0.7
                    event_log = read_log(
                        str(log_path / encoder / f"{log.removesuffix('.xes')}_{alg}_cf_{weight}_Crisp_decl.csv"))
                    alignments_bk = pm4py.fitness_token_based_replay(event_log, net, initial_marking, final_marking,
                                                                     activity_key=act_name_key,
                                                                     case_id_key=case_name_key,
                                                                     timestamp_key=timestamp_key)
                    encoder_results.append(alignments_bk['log_fitness'])
            writer.writerow([log + encoder] + [round(res,3) for res in encoder_results])




