import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_bars_w_old(plots_folder, file_name):
    df = pd.read_csv(file_name, delimiter=',')
    grouped = df.groupby(['weight'])
    total_counts = grouped.size().reset_index(name='total_count')
    total_counts['total_count']=total_counts['total_count']
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.histplot(data=total_counts, x='weight',bins=5, weights='total_count',color='skyblue')
    ax.set_xticks([0.5,0.6,0.7,0.8,0.9])
    plt.xlabel('weight (w)',fontsize=14)
    plt.ylabel(' ', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(plots_folder, f"{file_name.stem}.pdf"))


# PLOT 1: Weight

def plot_bars_w(plots_folder, file_name):
    df = pd.read_csv(file_name, delimiter=',')
    grouped = df.groupby(['weight'])
    total_counts = grouped.size().reset_index(name='total_count')
    total_counts['percentage'] = (total_counts['total_count'] / total_counts['total_count'].sum()) * 100

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(data=total_counts,x='weight', y='percentage', color='skyblue', ax=ax)
    plt.xlabel('Weight', fontsize=10)
    plt.ylabel('Percentage (%)', fontsize=10)
    plt.ylim(0, 100)

    for i, row in total_counts.iterrows():
        ax.text(i, row['percentage'] + 1, f"{row['percentage']:.1f}%", ha='center', fontsize=10)

    plt.subplots_adjust(left=0.2, right=0.95)
    plt.savefig(os.path.join(plots_folder, f"{file_name.stem}.pdf"), bbox_inches='tight')
    plt.close()