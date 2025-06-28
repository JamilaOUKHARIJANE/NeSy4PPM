import argparse
from pathlib import Path
from NeSy4PPM.Data_preprocessing import log_utils
from NeSy4PPM.Data_preprocessing.utils import NN_model, Encodings, load_bk
from NeSy4PPM.Prediction.predict_suffix import predict_evaluate
from NeSy4PPM.Training.train_model import learn

parser = argparse.ArgumentParser()
parser.add_argument('--train_log', default=None, help='train log', type=str)
parser.add_argument('--test_log', default=None, help='test log', type=str)
parser.add_argument('--encoder', default=None, help='encoder')
parser.add_argument('--bk_file', default=None, help='BK file name', type=str)

args = parser.parse_args()
bk_file_name = str(args.bk_file)
train_log = str(args.train_log)
test_log = str(args.test_log)
encoder_str = str(args.encoder)
if encoder_str=='one_hot':
    encoder = Encodings.One_hot
elif encoder_str=='index':
    encoder = Encodings.Index_based
elif encoder_str=='shrinked':
    encoder = Encodings.Shrinked_based
elif encoder_str=='multi':
    encoder = Encodings.Multi_encoders
bk_file_path = Path.cwd().parent/'data'/'input'/'declare_models'/bk_file_name

log_path = Path.cwd().parent/'data'/'input'/'logs'
model = NN_model.Transformer
model_folder= Path.cwd().parent/'data'/'output'

log_data = log_utils.LogData(log_path=log_path,train_log=train_log,test_log=test_log,resource=True)
learn(log_data, encoder, model_arch=model, resource=True, output_folder=model_folder)

(log_data.evaluation_prefix_start, log_data.evaluation_prefix_end) = (1,4)
model_arch = NN_model.Transformer
output_folder= Path.cwd().parent/'data'/'output'
beam_size = 3

#Greedy search (data-driven)
predict_evaluate(log_data, model_arch=model_arch, encoder=encoder,
                            output_folder=output_folder,beam_size=1,resource=True,  weight=0.0)
#Beam search (data-driven)
predict_evaluate(log_data, model_arch=model_arch, encoder=encoder,
                            output_folder=output_folder, beam_size=beam_size,resource=True,  weight=0.0)

#Beam search + BK-contextualized
bk_model = load_bk(bk_file_path)
weight=0.9
predict_evaluate(log_data, model_arch=model_arch, encoder=encoder,
                            output_folder=output_folder, bk_model=bk_model, beam_size=beam_size, resource=True, weight=weight, bk_end=False)

#Beam search (data-driven)+ BK_filter
bk_model = load_bk(bk_file_path)
predict_evaluate(log_data, model_arch=model_arch, encoder=encoder,
                            output_folder=output_folder, bk_model=bk_model, beam_size=beam_size, resource=True, weight=0.0,bk_end=True)

