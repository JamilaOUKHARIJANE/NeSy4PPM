import argparse
import statistics as stat
import tensorflow as tf
from numba.cuda import gpus

from src.commons import log_utils, shared_variables as shared
from src.evaluation import evaluation
from src.training import train_model


class ExperimentRunner:
    def __init__(self, train, evaluate, use_variant_split):
        self._train = train
        self._evaluate = evaluate
        self._use_variant_split = use_variant_split

        print('Perform training:', self._train)
        print('Perform evaluation:', self._evaluate)
        print('Use variant-based split:', self._use_variant_split)

    def run_experiments(self, log_list, test_log_list, alg, method_fitness, weight, resource, timestamp, outcome, model_folder):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth for each GPU to avoid consuming all memory upfront
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPUs detected and memory growth enabled.")
            except RuntimeError as e:
                print(e)
        if test_log_list is not None:
            for log_name, test_log_name in zip(log_list,test_log_list):
                self._run_single_experiment(log_name,test_log_name, alg, method_fitness, weight, resource, timestamp, outcome,
                                            model_folder)
        else:
            for log_name in log_list:
                self._run_single_experiment(log_name,None, alg, method_fitness, weight, resource, timestamp, outcome,model_folder)

    def _run_single_experiment(self, log_name, test_log_name, alg, method_fitness, weight, resource, timestamp, outcome,model_folder):
        log_data = log_utils.LogData(log_name,test_log_name ,self._use_variant_split, resource)
        log_data.encode_log(resource, timestamp, outcome)

        trace_sizes = list(log_data.log.value_counts(subset=[log_data.case_name_key], sort=False))

        print('Log name:', log_data.log_name.value + log_data.log_ext.value)
        print('Log size:', len(trace_sizes))
        print('Trace size avg.:', stat.mean(trace_sizes))
        print('Trace size stddev.:', stat.stdev(trace_sizes))
        print('Trace size min.:', min(trace_sizes))
        print('Trace size max.:', max(trace_sizes))
        print(f'Evaluation prefix range: [{log_data.evaluation_prefix_start}, {log_data.evaluation_prefix_end}]')
        log_data.maxlen = max(trace_sizes)

        if self._train:
            train_model.train(log_data, model_folder, resource, outcome) #keras_trans or LSTM
        
        if self._evaluate :
            evaluation.evaluate_all(log_data, model_folder, alg, method_fitness, weight, resource, timestamp,
                                    outcome)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--algo', default="beamsearch", help='use baseline or beamsearch', type=str)
    parser.add_argument('--model',default="keras_trans", help='use LSTM or Keras_trans', type=str)
    parser.add_argument('--use_Prob_reduction', default=False, action='store_true',help='use probability reduction')
    parser.add_argument('--beam', default=None, help='define beam size', type=str)
    parser.add_argument('--BK_weight', type=float, nargs='+', default=[0.0], help='set the weight of the background knowledge')
    parser.add_argument('--resource', default=False, action='store_true',help='predict resource')
    parser.add_argument('--encoding', default="index-based",
                       help='use one-hot, index-based, shrinked index-based or multi-encoders')
    parser.add_argument('--pipeline', default='evaluate', help='use train, evaluate or full_run')
    #parser.add_argument('--prefix', default=3, help='evaluation prefix start')
    parser.add_argument('--BK_end', default=False, action='store_true',
                       help='use background knowledge at the end of the prediction')
    parser.add_argument('--Decl_BK', default=None, type=str, help='use Prob_decl or Crisp_decl')
    parser.add_argument('--method_fitness', default=None, type=str,
                        help='use fitness_token_based_replay or fitness_alignments ')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test_log', default=None, help='input test log')
    group.add_argument('--use_variant_split', default=False, action='store_true', help='Use variant-based split for train/test')

    #group = parser.add_mutually_exclusive_group()

    args = parser.parse_args()

    logs = [args.log.strip()] if args.log else shared.log_list
    test_logs= [args.test_log.strip()] if args.test_log else None
    w = args.BK_weight
    Decl_BK = args.Decl_BK
    shared.declare_BK = True if w != [0.0] and Decl_BK == 'Crisp_decl' else False
    shared.prob_declare_BK = True if w != [0.0] and Decl_BK == 'Prob_decl' else False
    method_fitness = args.method_fitness if w != [0.0] else None #and Decl_BK is None and not args.BK_end
    shared.BK_end = args.BK_end
    #shared.prefix_start = int(args.prefix)
    if Decl_BK is not None:
        shared.BK_version=(shared.BK_version or '') + Decl_BK
    if method_fitness is not None:
        shared.BK_version=(shared.BK_version or '') + "Petri_net"
    if shared.BK_end:
        shared.BK_version=(shared.BK_version or '') + "BK_END"

    if args.encoding =='one-hot':
        shared.One_hot_encoding = True
    elif args.encoding == 'shrinked index-based':
        shared.combined_Act_res =True
    elif args.encoding =='multi-encoders':
        shared.use_modulator=True

    shared.useProb_reduction = args.use_Prob_reduction

    if args.beam:
        shared.beam_size = int(args.beam)

    if args.pipeline=='full_run':
        args.train = True
        args.evaluate = True
    elif args.pipeline=='evaluate':
        args.train = False
        args.evaluate = True
    elif args.pipeline=='train':
        args.train = True
        args.evaluate = False        
    

    ExperimentRunner(train=args.train,
                     evaluate=args.evaluate,
                     use_variant_split=args.use_variant_split) \
        .run_experiments(log_list=logs,
                         test_log_list =test_logs,
                         alg = args.algo,
                         method_fitness = method_fitness,
                         weight = w,
                         resource = args.resource,
                         timestamp = False,
                         outcome = False,
                         model_folder= args.model)
