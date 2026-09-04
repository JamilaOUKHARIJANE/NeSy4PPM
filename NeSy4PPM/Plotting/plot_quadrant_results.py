from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "SDFA_results"
OUT_BASE = RESULTS_DIR / f"benchmark_quadrant"
ALL_LOGS_PDF = OUT_BASE.with_name(OUT_BASE.name + "_all_logs.pdf")
PROCEDURAL_INPUT = RESULTS_DIR / f"procedural_results.csv"
DECLARATIVE_INPUT = RESULTS_DIR / f"declarative_results.csv"

DATASET_ORDER = [
    "BPIC2012",
    "BPI2013_In",
    "BPI2013_CP",
    "log",
    "helpdesk",
    "DomesticDeclarations",
    "InternationalDeclarations",
    "PermitLog",
    "PrepaidTravelCost",
    "RequestForPayment",
]
DATASET_LABELS = {
    "BPIC2012": "BPIC2012",
    "BPI2013_In": "BPIC2013-I",
    "helpdesk": "Helpdesk",
    "log": "Dreyers",
    "BPI2013_CP": "BPIC2013-CP",
    "PrepaidTravelCost": "Prepaid",
    "RequestForPayment": "Request",
    "PermitLog": "Permit",
    "InternationalDeclarations": "International",
    "DomesticDeclarations": "Domestic",
}

PANEL_CONFIGS = [
    {
        "name": "Procedural", #knowledge
        "input": PROCEDURAL_INPUT,
        "methods": [
            "NN",
            "DIFF-ERO","DIFF-ERO_global",
            "Petrinet","SDFA_end",
            "SDFA_noprune", "SDFA"
        ],
    },
    {
        "name": "Declarative", #knowledge
        "input": DECLARATIVE_INPUT,
        "methods": [
            "NN",
            "LLL", "GLL",
            "Declare","SDFA_declare_end",
            "SDFA_declare_noprune","SDFA_declare",

        ],
    },
]

METHOD_LABELS = {
    "NN": r"$\mathbf{NN}$",
    "DIFF-ERO": r"$\mathbf{DIFF\text{-}ERO}_{\mathrm{L}}$",
    "DIFF-ERO_global": r"$\mathbf{DIFF\text{-}ERO}_{\mathrm{G}}$",
    "SDFA": r"$\mathbf{SDFA}$",
    "SDFA_noprune": r"$\mathbf{SDFA}_{\mathrm{NP}}$",
    "Petrinet": r"$\mathbf{NN\otimes\mathcal{BK}}$",
    "PetriNet_end": r"$\mathbf{NN\rightarrow\mathcal{BK}}$",
    "Declare": r"$\mathbf{NN\otimes\mathcal{BK}}$",
    "Declare_end": r"$\mathbf{NN\rightarrow\mathcal{BK}}$",
    "LLL": r"$\mathbf{NN}_{\mathrm{LLL}}$",
    "GLL": r"$\mathbf{NN}_{\mathrm{GLL}}$",
    "SDFA_declare": r"$\mathbf{SDFA}$",
    "SDFA_declare_noprune": r"$\mathbf{SDFA}_{\mathrm{NP}}$",
"SDFA_declare_end":  r"$\mathbf{NN\rightarrow SDFA}$",
"SDFA_end":  r"$\mathbf{NN\rightarrow SDFA}$",
}

METHOD_COLOR_GROUPS = {
    "NN": "NN",
    "DIFF-ERO": "local",
    "DIFF-ERO_global": "global",
    "SDFA": "SDFA",
    "SDFA_declare": "SDFA",
    "SDFA_noprune": "SDFA_noprune",
    "SDFA_declare_noprune": "SDFA_noprune",
    "Petrinet": "BK",
    "Declare": "BK",
    #"PetriNet_end": "BK_end",
    #"Declare_end": "BK_end",
    "LLL": "local",
    "GLL": "global",
    "SDFA_end": "SDFA_end",
    "SDFA_declare_end": "SDFA_end",
}

COLOR_GROUP_ORDER = [
    "NN",
    "local",
    "global",
    "SDFA",
    "SDFA_noprune",
    "BK",
   # "BK_end",
    "SDFA_end",
]

# Fixed, colorblind-friendly palette. Equivalent procedural and declarative
# methods use the same color through METHOD_COLOR_GROUPS.
COLOR_GROUP_COLORS = {
    "NN": "green",
    "local": "brown",
    "global": "magenta",
    "SDFA": "blue",
    "SDFA_noprune": "red",
    "BK": "purple",
    "SDFA_end": "orange",
}

REQUIRED_COLUMNS = [
    "method",
    "Log",
    "Damerau-Levenshtien similarity",
    "Time",
    "EBC",
    "feasibility_rate",
    "termination_rate",
]
METRIC_COLUMNS = ["Time score", "EBC score","Feasibility", "Termination", "DLS"]
METRIC_LABELS = ["1 - nTime", "1 - nEBC","Feasibility", "Termination", "nDLS"]


def load_panel_data(config):
    df = pd.read_csv(config["input"])
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {config['input']}: {missing}")
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(f"Missing values found in required columns for {config['input']}.")

    numeric_columns = [
        "Damerau-Levenshtien similarity",
        "Time",
        "EBC",
        "feasibility_rate",
        "termination_rate",
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df["Dataset"] = df["Log"].str.replace(r"_\d+$", "", regex=True)

    agg = (
        df.groupby(["Dataset", "method"], as_index=False)
        .agg(
            DLS=("Damerau-Levenshtien similarity", "mean"),
            Time=("Time", "mean"),
            EBC=("EBC", "mean"),
            Feasibility=("feasibility_rate", "mean"),
            Termination=("termination_rate", "mean"),
        )
    )
    compared = agg[
        agg["Dataset"].isin(DATASET_ORDER) & agg["method"].isin(config["methods"])
    ].copy()
    compared["Knowledge"] = config["name"]
    return compared


def add_scores(panel_data, min_time, max_time):
    parts = []
    for config, compared in panel_data:
        for dataset in DATASET_ORDER:
            part = compared[compared["Dataset"] == dataset].copy()
            if set(part["method"]) != set(config["methods"]):
                raise ValueError(f"Unexpected method coverage for {config['name']} / {dataset}.")
            if max_time == min_time:
                part["Time score"] = 1.0
            else:
                part["Time score"] = (
                    1 - ((part["Time"] - min_time) / (max_time - min_time)).clip(0, 1)
                )
            part["EBC score"] = 1 - part["EBC"]
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def aggregate_all_datasets(scores):
    return (
        scores.groupby(["Knowledge", "method"], as_index=False)
        .agg(
            DLS=("DLS", "mean"),
            Time=("Time", "mean"),
            EBC=("EBC", "mean"),
            Feasibility=("Feasibility", "mean"),
            Termination=("Termination", "mean"),
            **{
                "Time score": ("Time score", "mean"),
                "EBC score": ("EBC score", "mean"),
            },
        ).round(3)
        .assign(Dataset="All datasets")
    )


panel_data = [(config, load_panel_data(config)) for config in PANEL_CONFIGS]
all_compared = pd.concat([data for _, data in panel_data], ignore_index=True)
min_time = all_compared["Time"].min()
max_time = all_compared["Time"].max()

scores = add_scores(panel_data, min_time, max_time)
all_datasets = aggregate_all_datasets(scores)
values_path = OUT_BASE.with_name(OUT_BASE.name + "_values.csv")
all_datasets[
    [
        "Knowledge",
        "Dataset",
        "method",
        "DLS",
        "Time",
        "EBC",
        "Feasibility",
        "Termination",
        "Time score",
        "EBC score",
    ]
].to_csv(values_path, index=False)

angles = np.linspace(0, 2 * np.pi, len(METRIC_COLUMNS), endpoint=False).tolist()
angles += angles[:1]

if set(COLOR_GROUP_ORDER) != set(COLOR_GROUP_COLORS):
    raise ValueError("COLOR_GROUP_COLORS must define every color group exactly once.")

color_group_map = {
    group: COLOR_GROUP_COLORS[group]
    for group in COLOR_GROUP_ORDER
}
color_map = {
    method: color_group_map[METHOD_COLOR_GROUPS[method]]
    for config in PANEL_CONFIGS
    for method in config["methods"]
}

def plot_quadrant(ax, plot_data, dataset, config, compact=False):
    """Plot one knowledge configuration for one dataset on ``ax``."""
    part = plot_data[
        (plot_data["Knowledge"] == config["name"])
        & (plot_data["Dataset"] == dataset)
    ].set_index("method")
    if set(part.index) != set(config["methods"]):
        raise ValueError(
            f"Unexpected method coverage for {config['name']} / {dataset}."
        )

    label_size = 15 if compact else 14
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=label_size)
    ax.tick_params(axis="x", pad=19 if compact else 19)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14
    )
    ax.set_rlabel_position(18)
    ax.grid(linewidth=0.65, alpha=0.62)
    ax.spines["polar"].set_linewidth(0.8)

    handles = []
    for method in config["methods"]:
        vals = part.loc[method, METRIC_COLUMNS].astype(float).tolist()
        vals += vals[:1]
        line, = ax.plot(
            angles,
            vals,
            linewidth=1.35 if compact else 1.6,
            marker="o",
            markersize=2.6 if compact else 3.2,
            color=color_map[method],
            label=METHOD_LABELS[method],
        )
        ax.fill(angles, vals, color=color_map[method], alpha=0.055)
        handles.append(line)
    return handles


def combined_legend_entries(legend_handles):
    """Return one deduplicated legend covering both knowledge configurations."""
    combined_handles = []
    combined_labels = []
    for config_index, config in enumerate(PANEL_CONFIGS):
        for handle, method in zip(legend_handles[config_index], config["methods"]):
            if METHOD_COLOR_GROUPS[method] == "local":
                label = (
                    r"$\mathbf{DIFF\text{-}ERO}_{\mathrm{L}}/"
                    r"\mathbf{NN}_{\mathrm{LLL}}$"
                )
            elif METHOD_COLOR_GROUPS[method] == "global":
                label = (
                    r"$\mathbf{DIFF\text{-}ERO}_{\mathrm{G}}/"
                    r"\mathbf{NN}_{\mathrm{GLL}}$"
                )
            else:
                label = METHOD_LABELS[method]
            if label not in combined_labels:
                combined_handles.append(handle)
                combined_labels.append(label)
    return combined_handles, combined_labels


def create_quadrant_figure(plot_data, dataset, show_page_title=False):
    """Create the paired procedural/declarative quadrant figure for one dataset."""
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8.2), subplot_kw={"projection": "polar"}
    )

    legend_handles = []
    for ax, config, panel_label in zip(axes, PANEL_CONFIGS, ["(a)", "(b)"]):
        handles = plot_quadrant(ax, plot_data, dataset, config)
        legend_handles.append(handles)

        ax.set_title(
            f"{panel_label} {config['name']}",
            y=-0.1,
            fontsize=26,
            fontweight="bold",
        )

    combined_handles, combined_labels = combined_legend_entries(legend_handles)
    fig.legend(
        combined_handles,
        combined_labels,
        title="Compared methods:",
        loc="center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=7,
        frameon=True,
        fontsize=18,
        title_fontsize=20,
    )

    fig.subplots_adjust(
        top= 0.94,
        bottom=0.28,
        left=0.04,
        right=0.98,
        wspace=0.04,
    )
    return fig


def create_all_logs_figure(plot_data):
    """Create one page with two quadrant subplots for every dataset."""
    fig, axes = plt.subplots(
        5,
        4,
        figsize=(16, 43),
        subplot_kw={"projection": "polar"},
    )
    legend_handles = [None] * len(PANEL_CONFIGS)
    panel_labels = ["(a)", "(b)"]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        row = dataset_index // 2
        pair_start_column = (dataset_index % 2) * len(PANEL_CONFIGS)
        for config_index, config in enumerate(PANEL_CONFIGS):
            column = pair_start_column + config_index
            ax = axes[row, column]
            handles = plot_quadrant(ax, plot_data, dataset, config, compact=True)
            ax.set_title(
                f"{panel_labels[config_index]} {config['name']}",
                y=-0.27,
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
            )
            if config_index == 0:
                # Center one log name over its procedural/declarative pair.
                ax.text(
                    1.08,
                    1.18,
                    DATASET_LABELS[dataset],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    fontweight="bold",
                )
            if legend_handles[config_index] is None:
                legend_handles[config_index] = handles

    combined_handles, combined_labels = combined_legend_entries(legend_handles)

    fig.legend(
        combined_handles,
        combined_labels,
        title="Compared methods:",
        loc="center",
        bbox_to_anchor=(0.5, -0.009),
        ncol=7,
        frameon=True,
        fontsize=14,
        title_fontsize=15,
    )

    fig.subplots_adjust(
        top=0.5, bottom=0.01, left=0.04, right=0.98, hspace=0.2, wspace=0.55
    )
    return fig


fig = create_quadrant_figure(all_datasets, "All datasets")
for ext in ("png", "pdf"):
    fig.savefig(
        OUT_BASE.with_suffix("." + ext),
        dpi=350 if ext == "png" else None,
        bbox_inches="tight",
    )
plt.close(fig)

# Store all logs on one page, with procedural and declarative quadrants side by side.
fig = create_all_logs_figure(scores)
fig.savefig(ALL_LOGS_PDF, bbox_inches="tight")
plt.close(fig)

print("Created outputs:")
for ext in ("png", "pdf"):
    print(OUT_BASE.with_suffix("." + ext))
print(ALL_LOGS_PDF)
print(values_path)
