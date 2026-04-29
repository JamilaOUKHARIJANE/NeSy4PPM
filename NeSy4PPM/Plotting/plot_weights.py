import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

folds=3

def aggregate_dls_weights(data, output_folder):
    results = {}
    for fold in range(folds):
        path = os.path.join(output_folder, str(fold), "results", "CF")
        list_data = [os.path.join(path, f) for f in os.listdir(path) if data in f]
        list_data.sort()
        dls_df = pd.concat([pd.read_csv(f) for f in list_data], ignore_index=True)  # Combine all CSVs
        average_dls = dls_df.groupby(['Weight', 'Prefix length'])['Damerau-Levenshtein Acts'].mean().reset_index()
        count_dls = dls_df.groupby(['Weight', 'Prefix length'])['Damerau-Levenshtein Acts'].count().reset_index()
        count_dls['x'] = count_dls['Damerau-Levenshtein Acts'] / count_dls.loc[
            count_dls['Weight'] == 0, 'Damerau-Levenshtein Acts'].sum()

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

def plot_dls_weights(dataset_results, output_folder,title):
    plt.rcParams.update({
        'font.size': 32,
        'axes.titlesize': 32,
        'axes.labelsize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28
    })
    f, ax = plt.subplots(6, 2, figsize=(28, 32))
    axes = ax.flatten()
    for idx, dataset in enumerate(dataset_results.keys()):
        results = dataset_results[dataset]
        subplot = axes[idx]
        subplot.set_title(dataset, fontweight='bold')
        subplot.set_xlabel('Weight')
        subplot.set_ylabel('Similarity')
        subplot.grid(True)
        subplot.set_xticks(np.arange(0, 1.01, 0.1))
        subplot.tick_params(axis='x', rotation=45)
        subplot.set_ylim(0, results['x'].max() + 0.2)
        sns.lineplot(data=results, x='Weight', y='x', marker='o', ax=subplot)
        for i, row in results.iterrows():
            if i%2 == 0:
                subplot.text(row['Weight']-0.02 , row['x'] + 0.02, f"{row['x']:.2f}", fontsize=28)
            else:
                subplot.text(row['Weight'] , row['x'] - 0.02, f"{row['x']:.2f}", fontsize=28, ha='center', va='top')

    # Hide any remaining unused subplots
    for ax_unused in axes[len(dataset_results):]:
        ax_unused.set_visible(False)

    plt.tight_layout()
    plot_dir = output_folder.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"{title}.pdf", bbox_inches="tight")
    plt.close()

def plots_dls_weights_pdf(loglist, output_folder, title):
    dataset_results = {}
    for log_name in loglist:
        dataset_results[log_name] = aggregate_dls_weights(f"{log_name}_feedback", output_folder)
    plot_dls_weights(dataset_results, output_folder,title)