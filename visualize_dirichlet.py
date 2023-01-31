import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import pandas as pd

# Visualizing the distribution of labels on clients
chart_folder = "thesis_chart_results"
root_folder = "to_visualize_distrib"
split = "train"

heatmap = "sattler"
# heatmap = "regular"
num_classes = 100
num_clients = 100

dirs = os.listdir(root_folder)
print(dirs)
plt.style.use("seaborn-whitegrid")
for d in dirs:
    path = os.path.join(root_folder, d, "distribution_" + split + ".npy")
    smpls_loaded = np.load(path)
    print(smpls_loaded)
    print(tf.reduce_sum(smpls_loaded))
    df = pd.DataFrame({})
    for label in range(0, num_classes):
        df[str(label)] = smpls_loaded[:, label]
    df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
    print(df.head())
    # print(df)
    if heatmap == 'sattler':
        g = sns.scatterplot(
            data=df, x="index", y="variable", size="value",
            sizes=(np.min(smpls_loaded), np.max(smpls_loaded)),
        )
    else:
        df = df.pivot("index", "variable", "value")

        g = sns.heatmap(data=df, cmap="Blues",
                        vmin=np.min(smpls_loaded), vmax=np.max(smpls_loaded),
                        yticklabels=10,
                        # xticklabels=10,
                        )
    # g.set_title("α = " + d, fontsize=16, pad=20))
    g.set_ylabel('Label', fontsize=16)
    g.set_xlabel('Client', fontsize=16)
    if heatmap == 'sattler':
        xmin, xmax = 0, 99
        tick_pos = np.linspace(xmin, xmax, 11)
        tick_labels = [h * 10 for h in range(len(tick_pos))]
        g.set_xticks(tick_pos)
        # plt.xticks(np.arange(0, num_clients + 1, round(num_clients / 10.0)))
        g.set_xticklabels([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        ymin, ymax = 0, 99
        tick_pos = np.linspace(ymin, ymax, 11)
        tick_labels = [h*10 for h in range(len(tick_pos))]
        g.set_yticks(tick_pos)
        g.set_yticklabels(reversed([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        # plt.yticks([i for i in reversed(range(0, num_clients + 1, round(num_clients / 10.0)))])
        # plt.legend(bbox_to_anchor=(1.25, 1.0), borderaxespad=0)
        # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title="Number of examples", title_fontsize=11, labelspacing=0.9, borderaxespad=0.8)
        sns.move_legend(
            g, "lower center",
            bbox_to_anchor=(.5, 1), ncol=6, title=None, frameon=False,
        )
        sns.despine(top=True, right=True, left=True, bottom=True)
        g.set_title("α = " + d, fontsize=16, pad=40)
        # g.get_legend().remove()
    else:
        xmin, xmax = g.get_xlim()
        tick_pos = np.linspace(xmin, xmax, 11)
        tick_labels = [h*10 for h in range(len(tick_pos))]
        g.set_xticks(tick_pos)
        g.set_xticklabels(tick_labels)

        ymin, ymax = g.get_ylim()
        tick_pos = np.linspace(ymin, ymax, 11)
        tick_labels = [h*10 for h in range(len(tick_pos))]
        g.set_yticks(tick_pos)
        g.set_yticklabels(tick_labels, rotation=0)
        g.set_title("α = " + d, fontsize=16, )

    plt.show()
    g.get_figure().savefig(os.path.join(chart_folder, 'distrib_' + d + '_' +heatmap+'.pdf'), format='pdf', bbox_inches='tight')
