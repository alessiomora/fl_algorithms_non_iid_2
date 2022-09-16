""" This script partions the CIFAR10 dataset in a federated fashion.
The level of non-iidness is defined via the alpha parameter (alpha in the paper below as well)
for a dirichlet distribution, and rules the distribution of examples per label on clients.
This implementation is based on the paper: https://arxiv.org/abs/1909.06335
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import shutil


def generate_dirichlet_samples(num_of_classes, alpha, num_of_clients,
                               num_of_examples_per_label):
    """Generate samples from a dirichlet distribution based on alpha parameter.
    Samples will have the shape (num_of_clients, num_of_classes).
    Returns an int tensor with shape (num_of_clients, num_of_classes)."""
    for _ in range(0, 10):
        alpha_tensor = tf.fill(num_of_clients, alpha)
        # alpha_tensor = alpha * prior_distrib
        # print(alpha_tensor)
        dist = tfp.distributions.Dirichlet(tf.cast(alpha_tensor, tf.float32))
        samples = dist.sample(num_of_classes)
        # Cast to integer for an integer number of examples per label per client
        int_samples = tf.cast(tf.round(samples * num_of_examples_per_label), tf.int32)
        int_samples_transpose = tf.transpose(int_samples, [1, 0])
        # print("reduce_sum", tf.reduce_sum(int_samples_transpose, axis=1))
        correctly_generated = tf.reduce_prod(tf.reduce_sum(int_samples_transpose, axis=1))
        if tf.cast(correctly_generated, tf.float32) != tf.constant(0.0, tf.float32):
            break
        print("Generated some clients without any examples. Retrying..")

    return int_samples_transpose


def remove_list_from_list(orig_list, to_remove):
    """Remove to_remove list from the orig_list and returns a new list."""
    new_list = []
    for element in orig_list:
        if element not in to_remove:
            new_list.append(element)
    return new_list


if __name__ == '__main__':
    alphas = [1.0] # alpha >= 100.0 generates a homogeneous distrib.
    datasets = ["cifar10"] # dataset = ["cifar10", "cifar100"]
    num_of_clients = 20

    print("Generating dirichlet partitions..")

    for dataset in datasets:
        for alpha in alphas:
            print("Generating alpha = "+ str(alpha) +" partitions..")
            # preparing folder
            folder = dataset + "_dirichlet"
            exist = os.path.exists(folder)

            if not exist:
                os.makedirs(folder)

            folder_split = str(round(alpha, 2))
            folder_path = os.path.join(folder, folder_split)
            exist = os.path.exists(folder_path)

            if not exist:
                os.makedirs(folder_path)
            else:
                shutil.rmtree(folder_path, ignore_errors=True)

            num_of_classes = 10 if dataset == "cifar10" else 100
            smpls = generate_dirichlet_samples(num_of_classes=num_of_classes, alpha=alpha,
                                               num_of_clients=num_of_clients,
                                               num_of_examples_per_label=5000 if dataset == "cifar10" else 500)
            # tf.print(smpls, summarize=-1)

            # print(tf.reduce_sum(smpls))
            # Loading the cifar10 dataset -- training split
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data() if dataset == "cifar10" \
                else tf.keras.datasets.cifar100.load_data()

            indexes_of_labels = list([list([]) for _ in range(0, num_of_classes)])

            j = 0
            for label in y_train:
                indexes_of_labels[label.item()].append(j)
                j = j + 1

            c = 0
            indexes_of_labels_backup = [element for element in indexes_of_labels]
            for per_client_sample in smpls:
                label = 0

                list_extracted_all_labels = []

                for num_of_examples_per_label in per_client_sample:
                    if len(indexes_of_labels[label]) < num_of_examples_per_label:
                        remained = len(indexes_of_labels[label])
                        extracted_1 = np.random.choice(indexes_of_labels[label], remained, replace=False)
                        indexes_of_labels[label] = indexes_of_labels_backup[label]
                        extracted_2 = np.random.choice(indexes_of_labels[label], num_of_examples_per_label - remained,
                                                       replace=False)
                        extracted = np.concatenate((extracted_1, extracted_2), axis=0)
                    else:
                        extracted = np.random.choice(indexes_of_labels[label], num_of_examples_per_label, replace=False)
                    indexes_of_labels[label] = remove_list_from_list(indexes_of_labels[label], extracted.tolist())

                    for ee in extracted.tolist():
                        list_extracted_all_labels.append(ee)

                    label = label + 1

                # print(list_extracted_all_labels)
                # print("list_extracted_all_labels", type(list_extracted_all_labels[0]))
                # if not list_extracted_all_labels:
                #     print("Empty list")
                # else:
                #     print("len ", len(list_extracted_all_labels))
                # for element in list_extracted_all_labels:
                #     if not(type(element) is int):
                #         print("Not int")
                #         print(type(element))
                #         print(element)
                #         element = int(element)
                list_extracted_all_labels = list(map(int, list_extracted_all_labels))
                numpy_dataset_y = y_train[list_extracted_all_labels]
                numpy_dataset_x = x_train[list_extracted_all_labels]

                ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
                ds = ds.shuffle(buffer_size=4096)

                tf.data.experimental.save(ds,
                                          path=os.path.join(os.path.join(folder_path, "train"),
                                                            str(c)))
                c = c + 1

            path = os.path.join(folder_path, "distribution.npy")
            np.save(path, smpls.numpy())
            smpls_loaded = np.load(path)
            # print(smpls_loaded)
