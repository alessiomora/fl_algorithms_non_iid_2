
# tensorboard dev upload --logdir logs
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorboard as tb
import os

alphas = [0.1]

chart_folder = "cifar10_chart_results"
experiment_id = "HY5r5sGdTPGx7YEBRKBglw"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
plt.style.use("seaborn-whitegrid")
df = experiment.get_scalars(pivot=False)
# dfw = dfw.loc[dfw['run'].str.contains('fedavg')]

# dfw = df.loc[df['run'].str.contains('|'.join(["fedavg", "feddf", "fedgkd", "best"]))]
dfw = df.loc[df['run'].str.contains('E20')]
dfw = dfw.loc[df['run'].str.contains('global')]
dfw = dfw.loc[df['tag'].str.contains('accuracy')]

for alpha in alphas:
    dfw_alpha = dfw.loc[df['run'].str.contains('cifar10_'+str(round(alpha, 2))+'/')]

    # df_hue = dfw.run.apply(lambda run: run.split("/")[0].replace("fedavg_", "α = "))
    df_hue = dfw_alpha.run.apply(lambda run: run.split("/")[1].split(",")[0])
    print(df_hue.head())

    print("---max---")
    #print(dfw.groupby(['run'], sort=False)['value'].max())
    df_max = dfw_alpha.groupby(['run'], sort=False, as_index=False)['value'].max()
    print(df_max)

    print("---std of max---")
    df_max['value'] = df_max['value'].apply(lambda val: val*100.0)
    df_max['run'] = df_max['run'].apply(lambda run: run.split(",")[0])
    print(df_max.groupby(['run'], sort=False).std())

    print("---average max---")
    print(df_max.groupby(['run'], sort=False).mean())
    #print("---max---")
    #print(df_hue.max())
    # print(optimizer_validation.head())


    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    g = sns.lineplot(data=dfw_alpha, x="step", y="value",
                 hue=df_hue)

    g.set_title("α = "+str(round(alpha, 2)), fontsize=14)
    g.set_ylabel('Accuracy', fontsize=14)
    g.set_xlabel('Round', fontsize=14)
    if alpha > 0.9:
        g.set_xlim([0, 100])
    g.legend(loc='lower right')
    g.get_legend().set_title(None)
    plt.setp(g.get_legend().get_texts(), fontsize='14')  # for legend text

    g.get_figure().savefig(os.path.join(chart_folder, 'accuracy_comparison_l'+
                                        str(round(alpha, 2))+'.pdf'), format='pdf', bbox_inches='tight')
    plt.show()