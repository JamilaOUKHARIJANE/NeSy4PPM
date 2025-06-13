import time
from multiprocessing import Process

import pm4py
import itertools
from pathlib import Path

from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessModels import DeclareModel
from pm4py.objects.log.util import dataframe_utils

from src.commons.log_utils import LogData
from src.commons import shared_variables as shared
from src.commons.utils import extract_bk_filename, extract_last_model_checkpoint, extract_Declare_bk_model, \
    extract_Prob_Declare_bk_model
from src.evaluation.prepare_data import prepare_encoded_data
from src.evaluation.inference_algorithms import beamsearch_cf, baseline_cf
from pm4py.algo.simulation.playout.petri_net import algorithm
from pm4py.algo.simulation.playout.petri_net.variants import extensive
from pm4py.algo.simulation.playout.petri_net.variants.extensive import Parameters
from pm4py.objects.conversion.log import converter

def evaluate_all(log_data: LogData, models_folder: str, alg: str, method_fitness: str, weight: list, resource: bool, timestamp: bool, outcome: bool):
    start_time = time.time()

    maxlen = log_data.maxlen
    predict_size = maxlen
    chars, chars_group, act_to_int, target_act_to_int, target_int_to_act,res_to_int, target_res_to_int, target_int_to_res \
        = prepare_encoded_data(log_data,resource)

    bk_filename = shared.pn_folder / (log_data.log_name.value + '_' + str(log_data.evaluation_prefix_start) + '.pnml')   # JH: .pnml
    #extract_bk_filename(log_data.log_name.value, log_data.evaluation_prefix_start)

    if (method_fitness == "conformance_diagnostics_alignments_prefix"):
        if 'bpmn' in str(bk_filename):
            bpmn = pm4py.read_bpmn(str(bk_filename))
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
            pm4py.write_pnml(net, initial_marking, final_marking, str(bk_filename).split(".")[0] + ".pnml")

        else:
            net, initial_marking, final_marking = pm4py.read_pnml(bk_filename)
            
        print("Start unfolding petrinet")
        sim_log = algorithm.apply(net, initial_marking, final_marking=final_marking, variant=extensive, parameters= {Parameters.MAX_TRACE_LENGTH: min(30, maxlen) })
        sim_data = pm4py.convert_to_dataframe(sim_log)
        print("Finished unfolding petrinet")

        prefix = list(sim_data.groupby('case:concept:name').apply(lambda x: list(range(1, len(x)+1))))
        prefix = list(itertools.chain(*prefix))
        sim_data['prefix'] = prefix
        
        for prefix_len in range(log_data.evaluation_prefix_start, predict_size+1):
            sim_data_prefix = sim_data.loc[sim_data['prefix'] < prefix_len+1].reset_index(drop= True)
            net_prefix, im_prefix, fm_prefix = pm4py.discover_petri_net_inductive(sim_data_prefix)
            pm4py.write_pnml(net_prefix, im_prefix, fm_prefix, str(bk_filename).split(".")[0] + "_" + str(prefix_len) + ".pnml")
    
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    compliant_traces = evaluation_traces

    tree = pm4py.discover_process_tree_inductive(evaluation_traces, noise_threshold = 0.0,  activity_key=log_data.act_name_key,
                                                        case_id_key=log_data.case_name_key,
                                                        timestamp_key= log_data.timestamp_key)    
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    
    pm4py.write_pnml(net, initial_marking, final_marking, bk_filename)

    print("Compliant traces: " + str(compliant_traces[log_data.case_name_key].nunique())
          + " out of " + str(len(log_data.evaluation_trace_ids)))
    print('Elapsed time:', time.time() - start_time)

    models_folder += "_One_hot" * (shared.One_hot_encoding and not shared.use_modulator) + \
                     "_Combined_Act_res" * (shared.combined_Act_res and not shared.use_modulator) + \
                     "_Multi_Enc" * (shared.use_modulator and not shared.One_hot_encoding) + \
                     "_Multi_One_hot_Enc" * (shared.use_modulator and shared.One_hot_encoding) + \
                     "_Simple_categorical" * (not shared.One_hot_encoding and not shared.combined_Act_res and not shared.use_modulator)

    # extract declare model
    bk_model = None
    if shared.declare_BK or shared.BK_end:
        '''log = D4PyEventLog()
        log.parse_xes_log(str(shared.log_folder) + f'/{log_data.test_log_name}.xes')
        disovery = DeclareMiner(log=log, consider_vacuity=False, min_support=1.0, itemsets_support=0.3, max_declare_cardinality=1)
        bk_model: DeclareModel = disovery.run() ''' #
        bk_model = extract_Declare_bk_model(log_data.test_log_name)
    elif shared.prob_declare_BK:
        bk_model = extract_Prob_Declare_bk_model(log_data.test_log_name)

    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        start_time = time.time()

        folder_path = shared.output_folder / models_folder / str(fold) / 'results_prody' / eval_algorithm
        if not Path.exists(folder_path):
            Path.mkdir(folder_path, parents=True)

        print(f"fold {fold} - {eval_algorithm}")
        if alg == "beamsearch":
            output_filename = folder_path / (f'{log_data.log_name.value if log_data.test_log_name is None else log_data.test_log_name}_beam{str(shared.beam_size)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}'
                                             f'{"Prob_reduction" * shared.useProb_reduction}_{shared.BK_version}.csv')

            print('beamsearch')
            model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF' + 'R'*resource + 'O'*outcome)
            beamsearch_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                            target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                            target_int_to_res, model_filename, output_filename, bk_filename, method_fitness, resource, outcome, weight, bk_model)
        elif alg=="baseline":
            output_filename = folder_path / f'{log_data.log_name.value if log_data.test_log_name is None else log_data.test_log_name}_{str(alg)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}.csv'

            print('baseline')
            model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF' + 'R'*resource + 'O'*outcome)
            baseline_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                            target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                            target_int_to_res, model_filename, output_filename, bk_filename, method_fitness, resource, outcome, weight)
        else:
            raise RuntimeError(f"No evaluation algorithm called: '{eval_algorithm}'.")

        print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))
