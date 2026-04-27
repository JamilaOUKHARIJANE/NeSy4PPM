import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pm4py
import seaborn as sns

from NeSy4PPM.Data_preprocessing.utils import Encodings


def aggregate_dls_weights(data, output_folder):
    results = {}
    for fold in range(3):
        path = os.path.join(output_folder, str(fold), "results1704", "CF")
        if data.startswith("Synthetic") or data.startswith("log") or data.startswith("PermitLog") or data.startswith("RequestForPayment")or (data.startswith("Prepaid") and encoder==Encodings.Index_based ):path = os.path.join(output_folder, str(fold), "results_time", "CF")
        list_data = [os.path.join(path, f) for f in os.listdir(path) if data in f]
        list_data.sort()
        dls_df = pd.concat([pd.read_csv(f) for f in list_data], ignore_index=True)  # Combine all CSVs
        average_dls = dls_df.groupby(['Weight', 'Prefix length'])['Damerau-Levenshtein Acts'].mean().reset_index()
        count_dls = dls_df.groupby(['Weight', 'Prefix length'])['Damerau-Levenshtein Acts'].count().reset_index()
        count_dls['x'] = count_dls['Damerau-Levenshtein Acts'] / count_dls.loc[
            count_dls['Weight'] == 0, 'Damerau-Levenshtein Acts'].sum()

        # Weighted average
        average_dls['x'] = average_dls['Damerau-Levenshtein Acts'] * count_dls['x']
        average_dls = average_dls.groupby('Weight')['x'].sum().reset_index()
        average_dls = average_dls[average_dls['Weight'] <= 1.0]
        average_dls['x'] = average_dls['x'].round(3)

        results[fold] = average_dls

    # Combine folds
    average_dls_folds = results[0].copy()
    average_dls_folds['x'] = (results[0]['x'] + results[1]['x'] + results[2]['x']) / 3
    average_dls_folds['x'] = average_dls_folds['x'].round(3)
    return average_dls_folds

def plot_dls_weights(dataset_results, output_folder):
    plt.rcParams.update({'font.size': 16})
    f, ax = plt.subplots(6, 2, figsize=(20, 20))
    axes = ax.flatten()

    for idx, dataset in enumerate(dataset_results.keys()):
        results = dataset_results[dataset]
        subplot = axes[idx]
        subplot.set_title(dataset, fontsize=18, fontweight='bold')
        subplot.set_xlabel('Weight', fontsize=18)
        subplot.set_ylabel('Similarity', fontsize=18)
        subplot.grid(True)
        subplot.set_xticks(np.arange(0, 1.05, 0.05))
        subplot.tick_params(axis='x', rotation=90, labelsize=16)
        subplot.tick_params(axis='y', labelsize=16)
        subplot.set_ylim(0, results['x'].max() + 0.2)
        sns.lineplot(data=results, x='Weight', y='x', marker='o', ax=subplot)
        for i, row in results.iterrows():
            if i%2 == 0:
                subplot.text(row['Weight']-0.02 , row['x'] + 0.02, f"{row['x']:.3f}", fontsize=14)
            else:
                subplot.text(row['Weight'] , row['x'] - 0.02, f"{row['x']:.3f}", fontsize=14, ha='center', va='top')

    # Hide any remaining unused subplots
    for ax_unused in axes[len(dataset_results):]:
        ax_unused.set_visible(False)

    plt.tight_layout()
    title = f"w_trend_feedback_results"+"_Index"
    plt.savefig(os.path.join(output_folder.parent / "plots", f'{title}.pdf'))
    plt.close()

def plots_dls_weights_pdf(loglist, output_folder):
    dataset_results = {}
    log_names= ["Synthetic-P","Synthetic-PS", "Sepsis",#'Road Traffic',
             'Helpdesk', "Dreyers", "BPIC2012",
                "BPIC2013 (I)", 'BPIC2013 (CP)', 'Prepaid Travel Cost',
                'Request For Payment','International Declarations', 'Domestic Declarations' #, "Permit Log"
    ]

    for log, log_name in zip(loglist,log_names):
        dataset_results[log_name] = aggregate_dls_weights(f"{log}_feedback", output_folder)
        #dataset_results[log_name] = aggregate_dls_weights(f"{log}test_prediction.csv", output_folder)
    plot_dls_weights(dataset_results, output_folder)

encoder = Encodings.Index_based
if encoder == Encodings.One_hot:
    output_folder = Path.cwd().parent.parent/"docs/source/data/Procedural/output/keras_trans_one-hot"
elif encoder == Encodings.Index_based:
    output_folder = Path.cwd().parent.parent/"docs/source/data/Procedural/output/keras_trans_index-based"
loglist = ["Synthetic1","Synthetic", "Sepsis_cases",#'Road_Traffic',
           'helpdesk',"log","BPIC2012", "BPI2013_In",'BPI2013_CP', 'PrepaidTravelCost',
           'RequestForPayment',  'InternationalDeclarations', 'DomesticDeclarations' #"PermitLog",
         ]
plots_dls_weights_pdf(loglist, output_folder)