
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorboard as tb
import os

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
# chart_folder = "cifar10_chart_results"
experiment_id = "Tbs4B4PkRJuY7SnTe7U5cA"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
plt.style.use("seaborn-whitegrid")
df = experiment.get_scalars(pivot=False)
print(df.head())
# dataset = "cifar10"
dataset = "cifar100"
alpha = 0.1
dfw = df[df['tag'].str.contains('accuracy')]
dfw = dfw[dfw['run'].str.contains(dataset+'_')]
dfw = dfw[dfw['run'].str.contains('|'.join(["2010", "2011", "2012"]))]

# dfw['run'] = dfw['run'].apply(lambda run: run.replace(run.split("/")[0], ""))
# dfw['run'] = dfw['run'].replace(r'E\d*_', '', regex=True)
# dfw['run'] = dfw['run'].apply(lambda run: run.replace("C5_total_clients100_lr_client0.1_momentum0.0_weight_decay0.0001_architectureresnet8_server_side_optimizersgd_lr_server1.0_server_momentum0.0_lr_decay0.998_", ""))
# dfw['run'] = dfw['run'].apply(lambda run: run.replace("/fedavg/,seed2018/global_test", ''))
# dfw['run'] = dfw['run'].apply(lambda run: run.replace(dataset+"_"+"alpha"+str(round(alpha, 2))+"_C20_k4/", ''))
dfw['run'] = dfw['run'].apply(lambda run: run.replace(dataset+"_"+"alpha"+str(round(alpha, 2))+"_C100_k5/", ''))
dfw['run'] = dfw['run'].apply(lambda run: run.replace('resnet8/', ''))
dfw['run'] = dfw['run'].apply(lambda run: run.replace('/global_test', ''))
dfw['run'] = dfw['run'].apply(lambda run: run.replace(run.split("/")[0], ''))
dfw['run'] = dfw['run'].apply(lambda run: run.replace("_server_side_optimizersgd_lr_server1.0_server_momentum0.0", ''))
dfw['run'] = dfw['run'].apply(lambda run: run.replace("_momentum0.0_", ''))

# dfw['run'] = dfw.groupby(['run'], sort=False, as_index=False)
idx = dfw.groupby(['run'])['value'].transform(max) == dfw['value']

print(dfw[idx])