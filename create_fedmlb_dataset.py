# importing the module
from dirichlet0_3_balanced_100C import data_mlb
import os
import tensorflow as tf
import numpy as np
import shutil
import sys

np.set_printoptions(threshold=sys.maxsize)

dataset = "cifar100"
folder = dataset + "_mlb_dirichlet_train_and_test"
exist = os.path.exists(folder)
alpha = 0.3
if not exist:
    os.makedirs(folder)

folder_split = str(round(alpha, 2))
folder_path = os.path.join(folder, folder_split)
exist = os.path.exists(folder_path)

if not exist:
    os.makedirs(folder_path)
else:
    shutil.rmtree(folder_path, ignore_errors=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() if dataset == "cifar10" \
    else tf.keras.datasets.cifar100.load_data()

for client in data_mlb:
    list_extracted_all_labels = data_mlb[client]
    numpy_dataset_y = y_train[list_extracted_all_labels]
    numpy_dataset_x = x_train[list_extracted_all_labels]

    ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
    ds = ds.shuffle(buffer_size=4096)

    tf.data.experimental.save(ds, path=os.path.join(os.path.join(folder_path, "train"), str(client)))


path = os.path.join(dataset+"_mlb_dirichlet_train_and_test", str(round(alpha, 2)), "train")


list_of_narrays = []
for sampled_client in range(0, 100):
    loaded_ds = tf.data.experimental.load(
        path=os.path.join(path, str(sampled_client)), element_spec=None, compression=None, reader_func=None
    )

    print("[Client "+str(sampled_client)+"]")
    print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds).numpy())

    def count_class(counts, batch, num_classes=100):
        _, labels = batch
        for i in range(num_classes):
            cc = tf.cast(labels == i, tf.int32)
            counts[i] += tf.reduce_sum(cc)
        return counts

    initial_state = dict((i, 0) for i in range(100))
    counts = loaded_ds.reduce(initial_state=initial_state, reduce_func=count_class)

    # print([(k, v.numpy()) for k, v in counts.items()])
    new_dict = {k: v.numpy() for k, v in counts.items()}
    # print(new_dict)
    res = np.array([item for item in new_dict.values()])
    # print(res)
    list_of_narrays.append(res)

distribution = np.stack(list_of_narrays)
print(distribution)
path = os.path.join(folder_path, "distribution_train.npy")
np.save(path, distribution)