import matplotlib.pyplot as plt
import os
from pathlib import Path
from NeSy4PPM.Data_preprocessing import shared_variables as shared

def add_plot(axs, metric, results):
    handles = []
    labels = []
    for i in results.keys():
        result_list = results[i][metric]
        prefix_length_list = results[i]["Prefix length"]
        line, =axs.plot(prefix_length_list, result_list,
                 color=shared.method_color[i],
                 marker=shared.method_marker[i],
                 label=i)
        handles.append(line)
        labels.append(i)
    return handles, labels

def plot_results(dataset_results,output_folder):
    for metric in ["Damerau-Levenshtein Acts", "Damerau-Levenshtein Resources"]:
        if not Path.exists(output_folder):
            Path.mkdir(output_folder, parents=True)

        f, ax = plt.subplots(8, 4, figsize=(16, 32))
        plt.rcParams.update({'font.size': 14})
        titles = ["", "Index-based", "Multi-Encoders", "Shrunk Index-based"]
        for i, dataset in enumerate(dataset_results.keys()):
            if i % 2 == 0:
                group = [ax[i][0], ax[i][1], ax[i+1][0], ax[i+1][1]]
            else:
                group = [ax[i-1][2], ax[i-1][3], ax[i][2], ax[i][3]]
            for j, (encoder, subplot) in enumerate(zip(dataset_results[dataset].keys(), group)):
                results = dataset_results[dataset][encoder]
                subplot.set_title(titles[j], fontsize=16, fontstyle='italic')
                subplot.set_xlabel('Prefix length (% of trace)', fontsize=14)
                subplot.set_ylabel(f'nDLS{" Res." if metric== "Damerau-Levenshtein Resources" else " Acts."}', fontsize=14)
                if j==0:
                    handles, labels = add_plot(subplot, metric, results)
                else:
                    handles, _ = add_plot(subplot, metric, results)
                subplot.grid()
            group[0].set_title(dataset, fontstyle='normal', fontsize=18, fontweight="bold", loc="left", pad=20)
            group[0].text(
                0.5, 1.0, "One-hot", fontstyle='italic', fontsize=16, ha='center', va='bottom', transform=group[0].transAxes
            )
        plt.tight_layout()
        f.subplots_adjust(top=0.96, bottom=0.05,hspace=0.4, wspace=0.3)
        # Create the legend
        legend = f.legend(labels=labels, title="Method", bbox_to_anchor=(0.5, 0.0), loc="lower center",
                          ncol=3, borderaxespad=0., title_fontsize='large', fontsize='large')

        # Set the title font weight to bold
        legend.get_title().set_fontweight('bold')

        # Save the figure
        title = f"average_{metric}_similarity_results"
        plt.savefig(os.path.join(output_folder, f'{title}.pdf'))
        plt.close()

def plot_results_encoder(dataset_results, encoder,output_folder):
    if not Path.exists(output_folder):
        Path.mkdir(output_folder, parents=True)

    f, ax = plt.subplots(4, 4, figsize=(18, 18))

    for subplot in ax.flatten():
        subplot.set_visible(False)

    plt.rcParams.update({'font.size': 18})
    titles = ["", "Resource"]
    k = 0
    for i, dataset in enumerate(dataset_results.keys()):
        if i % 2 == 0:
            group = [ax[i - k][0], ax[i - k][1]]
        else:
            group = [ax[i - 1 - k][2], ax[i - 1 - k][3]]
            k = k + 1
        for j, subplot in enumerate(group):
            subplot.set_visible(True)
            results = dataset_results[dataset][encoder]
            if j != 0:
                metric = "Damerau-Levenshtein Resources"
            else:
                metric = "Damerau-Levenshtein Acts"
            subplot.set_title(titles[j], fontsize=18, fontstyle='italic')
            subplot.set_xlabel('Prefix length', fontsize=18)
            subplot.set_ylabel(f'nDLS{" Res." if metric == "Damerau-Levenshtein Resources" else " Acts."}', fontsize=18)
            handles, labels = add_plot(subplot, metric, results)
            subplot.grid()
        group[0].set_title(dataset, fontsize=18, fontstyle='normal', fontweight="bold", loc="left", pad=20)
        # Add "Activity" label below the dataset title
        group[0].text(
            0.5, 1.0, "Activity", fontstyle='italic', fontsize=14, ha='center', va='bottom',
            transform=group[0].transAxes
        )
    plt.tight_layout()
    f.subplots_adjust(top=0.94, bottom=0.13, hspace=0.5, wspace=0.4)
    # Create the legend
    legend = f.legend(labels=labels, title="Method:", bbox_to_anchor=(0.5, 0.0), loc="lower center",
                      ncol=3, borderaxespad=0., title_fontsize='large', fontsize='large')
    legend.get_title().set_fontweight('bold')
    title = f"average_similarity_results_{encoder.name}"
    plt.savefig(os.path.join(output_folder, f'{title}.pdf'))
    plt.close()
