# Predictive process monitoring with online constraints
This repository contains the source code of a Neuro-symbolic predictive process monitoring system that provides:
- prediction of next activities of a process instance
- prediction of next activities with allocated resources of a process instance 

## Requirements
The following Python packages are required:

-   [keras]() tested with version 3.4.1;
-   [tensorflow]() tested with version 2.17.0;
-   [jellyfish]() tested with version 0.9.0;
-   [Distance]() tested with version 0.1.3;
-   [pm4py]() tested with version 2.5.2;
-   [declare4py]() tested with version 2.2.0;
-   [matplotlib](https://matplotlib.org/) tested with version 3.6.3;
-   [numpy]() tested with version 1.26.4;
-   [pandas]() tested with version 1.5.3;
-   [keras-nlp]() tested with version 0.14.0.


## Usage
The system has been tested with Python 3.10 After installing the requirements, please download this repository.

## Repository Structure
- `data/input` contains the input logs in`.xes`,`.xes.gz`or`.csv` format and the BPMN models of these logs;
- `media/output` contains the trained models and results of predictive process monitoring;
- `src/commons` contains the code defining the main settings for the experiments (training and evaluation);
- `src/training` contains the code for Neural Networks model training;
- `src/ProbDeclmonitor` contains the code for probabilistic declare monitoring;
- `src/evaluation` contains the code for evaluating the trained model and generating predictions; 
- `experiments_runner.py` is the main Python script for running the experiments;
- `results_aggregator.py` is a Python script for aggregating the results of each dataset and presenting in a more 
  understandable format.
- `plot_results.py` is a Python script for generating plots from the aggregated results.
  

## Running the code
### (1) Training
To train a Neural Networks model: **LSTM**: `--model="LSTM"` or **transformer** `--model="keras_trans"` for a 
given dataset (event log), type: 
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --pipeline="train"
```
#### (a) Encoding
The categorical data is used in training model as defined in the event log. 
By default, this data is encoded using Index-based encoding.
If you prefer other encoding methods, you can specify them using the `--encoding=` option:
- **One-hot encoding**. add `--encoding="one-hot"`:
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --pipeline="train" --encoding="one-hot"
```
- **Shrinked-Index-based**. use `--encoding="shrinked index-based"`:
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --pipeline="train" --encoding="shrinked index-based"
```
- **Multi-Encoders**. use `--encoding="multi_encoders"`
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --pipeline="train" --encoding="multi_encoders"
```
#### (a) Dataset splitting
By default, the given dataset is split in 80% for the training and 20% for the testing.

If you want to change the splitting strategy:
- **Variant-based splitting**. add `--use_variant_split` to split based on process variants:
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --pipeline="train" --use_variant_split
```
- **Using a separate test set**. to use an external test log, specify it with `--test_log` option:
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --pipeline="train" --test_log="helpdesk_test.xes"
```
### (2) Evaluation
To run the evaluation for a given (pretrained) dataset, you need to specify the prediction algorithm: baseline `--algo="baseline"` 
for simple autoregressive prediction or `--algo="beamsearch"` to use a [Beam Search](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24) algorithm:

```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --algo="baseline" --pipeline="evaluate"
```

Additionally, if you want to add the conformance checking of the Background Knowledge (BK) during the `beamsearch` algorithm 
to optimize the evaluation process, you need use the 'BK_weight' option that enables you to set the importance of the BK:

```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --algo="beamsearch" --pipeline="evaluate" --BK_weight 0.9
```
To specify the type of used BK, you can provide:

- `--Decl_BK="Crisp_decl"` for **(MP-) declare**:
```
python run_experiments.py --log='helpdesk.xes' --pipeline="evaluate" --test_log="helpdesk_test.xes" --BK_weight 0.9 --Decl_BK="Crisp_decl"
```
- `--Decl_BK="Prob_decl"` for **Probabilistic declare**:
```
python run_experiments.py --log='helpdesk.xes' --pipeline="evaluate" --test_log="helpdesk_test.xes" --BK_weight 0.9 --Decl_BK="Prob_decl"
```
- `--method_fitness` for **Petri Net BK**, where you need to specify the used fitness method:

```
python run_experiments.py --log='helpdesk.xes' --pipeline="evaluate" --test_log="helpdesk_test.xes" --BK_weight 0.9 --method_fitness="fitness_token_based_replay"
```
Additionally, you can use the `--BK_end` option to check the BK at the end of the prediction process.

```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --algo="beamsearch" --pipeline="evaluate" --BK_end
```

Moreover, if you want to minimise the prediction of redundant activities/resources, 
you need to apply the probability reduction for repetitive activities/resources
in a trace prefix by adding`--use_Prob_reduction`:
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --algo="baseline" --pipeline="evaluate" --use_Prob_reduction
```
### Training and evaluation
if you want to train and evaluate your model in the same experiment, you need to set the `full_run` option to pipeline instead of using `train` and then `evaluate` :
```
python run_experiments.py --log='helpdesk.xes' --model="keras_trans" --algo="baseline" --pipeline="full_run"
```

### Gathering the results
After running the experiments, type:
```
python results_aggregator.py 
```
to aggregate the Damerau-Levenshtein distance of activities and resources for all datasets. The results will be in the 
file`aggregated_results.csv` in `media/output`folder. 
