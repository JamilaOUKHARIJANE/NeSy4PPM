import matplotlib.pyplot as plt
import numpy as np

def build_trend(data, metric):
    return (
        data.groupby(["violation_rate", "method_label"], as_index=False)[metric]
            .mean()
            .sort_values(["method_label", "violation_rate"])
    )

def plot_bars_on_axis(subplot, log_data, metric, version_order, method_labels):
    trend = build_trend(log_data, metric)
    bar_data = (
        trend.pivot(index="violation_rate", columns="method_label", values=metric)
             .reindex(index=list(version_order.values()), columns=list(method_labels.values()))
    )
    x = np.arange(len(bar_data.index))
    methods = list(method_labels.values())
    bar_width = 0.8 / len(methods)

    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * bar_width
        subplot.bar(
            x + offset,
            bar_data[method],
            width=bar_width,
            label=method,
        )
    subplot.set_xticks(x)
    subplot.set_xticklabels(["0%", "10%", "25%", "50%"])
    subplot.set_ylim(0, 1.0)
    subplot.grid(True, axis="y", linestyle="--", alpha=0.5)


def plot_all_logs_bars_pdf(output_folder, title,method_labels, df, version_order,dataset_names=None):
    f, ax = plt.subplots(4, 4, figsize=(18, 18))

    for subplot in ax.flatten():
        subplot.set_visible(False)

    logs = df["dataset"].drop_duplicates().tolist()
    plt.rcParams.update({'font.size': 18})
    titles = ["", "Resource"]
    k = 0
    for i, dataset in enumerate(logs):
        if i % 2 == 0:
            group = [ax[i - k][0], ax[i - k][1]]
        else:
            group = [ax[i - 1 - k][2], ax[i - 1 - k][3]]
            k = k + 1
        for j, subplot in enumerate(group):
            subplot.set_visible(True)
            results = df[df["dataset"] == dataset]
            if j != 0:
                metric = "Damerau-Levenshtein Resources"
            else:
                metric = "Damerau-Levenshtein Acts"
            subplot.set_title(titles[j], fontsize=18, fontstyle='italic')
            subplot.set_xlabel('BK violation rate', fontsize=18)
            subplot.set_ylabel(f'nDLS{" Res." if metric == "Damerau-Levenshtein Resources" else " Acts."}', fontsize=18)
            plot_bars_on_axis(subplot, results, metric, version_order, method_labels)
            subplot.grid()
        group[0].set_title(dataset_names[dataset] if dataset_names else dataset, fontsize=18, fontstyle='normal', fontweight="bold", loc="left", pad=20)
        # Add "Activity" label below the dataset title
        group[0].text(
            0.5, 1.0, "Activity", fontstyle='italic', fontsize=14, ha='center', va='bottom',
            transform=group[0].transAxes
        )

    handles, labels = ax[0, 0].get_legend_handles_labels()
    f.legend(handles, labels, loc="upper center", ncol=len(method_labels), bbox_to_anchor=(0.5, 1.0))
    f.tight_layout(rect=[0, 0, 1, 0.96])
    f.savefig(output_folder/title, bbox_inches="tight")

