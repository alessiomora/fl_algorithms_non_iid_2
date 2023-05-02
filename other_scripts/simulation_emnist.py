"""This is a minimal implementation of Federated Distillation based on the papers:
https://arxiv.org/abs/2008.06180
https://ieeexplore.ieee.org/abstract/document/9435947
"""
# tensorboard dev upload --logdir /home/amora/pycharm_projects/fed_dist_simulated/exp_results/logs --name "Cifar10 8 clients" --description "Comparison of algorithms."
# https://tensorboard.dev/experiment/ULzAiK7SQWCE4LCox9tnUA/#scalars&regexInput=0.1%2Fglo
import os
import data_utility
import tensorflow as tf
import numpy as np
from tensorflow import keras
import residual_model as res
import itertools as it
import FedGKDModel as gkt
import FedProxModel as prox
import tensorflow_datasets as tfds
import functools
import logging_utility as logging


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        device=gpu, enable=True
    )

# import datetime

PATH = "emnist_100_results"

#def create_cnn_model(only_digits=False, determinism=True):
def create_cnn_model(only_digits=False):
    """The CNN model used in https://arxiv.org/abs/1602.05629.
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only EMNIST dataset. If False, uses 62 outputs for the larger
        dataset.
      dropout_rate: Base is 20%, can be changed inside application_paramaters.py
    Returns:
      An uncompiled `tf.keras.Model`.
    """
    data_format = 'channels_last'
    input_shape = [28, 28, 1]
    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu)

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10 if only_digits else 47),
    ])
    return model


def zip_img_and_pred(ds, teacher_pred, batch_size_distill, shuffle_size=1024):
    """Zip the input data with teacher predictions. Also shuffle and batch for training."""
    img_ds = ds.map(lambda image, label: image)
    zipped_img_pred = tf.data.Dataset.zip((img_ds, teacher_pred))
    zipped_img_pred = zipped_img_pred.shuffle(shuffle_size)
    zipped_img_pred = zipped_img_pred.batch(
        batch_size_distill)
    return zipped_img_pred


def element_norm_fn_emnist(image, label):
    """Utility function to normalize input images."""
    return tf.cast(image, tf.float32) / 255.0, label


if __name__ == '__main__':

    # Building a dictionary of hyperparameters
    hp = {}

    hp["algorithm"] = ["fedprox", "fedgkd"]
    # hp["algorithm"] = ["fedavg", "fedprox"]
    hp["batch_size"] = [16]
    hp["E"] = [10]  # local_epochs
    hp["C"] = [20]  # n clients
    hp["rounds"] = [50]
    hp["alpha"] = [0.05]
    hp["dataset"] = ["emnist"]
    hp["seed"] = [2021]  # seed for client selection

    # Creating a list of dictionaries
    # each one for a combination of hp
    cd = {}
    cd["fedprox"] = {"mu": [0.001]}
    cd["fedgkd"] = {"gamma": [0.001]}
    # Creating a list of dictionaries
    # each one for a combination of hp + algorithm-specific hyperparams
    settings = logging.get_combinations(hp, cd)

    # Running a simulation for each of the setting in hp possible combinations
    for setting in settings:
        print("simulation start " + str(setting))
        total_rounds = setting["rounds"]
        total_clients = setting["C"]
        local_epochs = setting["E"]  # local_epochs
        local_batch_size = setting["batch_size"]
        test_batch_size = 256
        alpha = setting["alpha"]
        algorithm = setting["algorithm"]
        random_seed = setting["seed"]
        dataset = setting["dataset"]

        num_classes = 10
        if dataset == "emnist":
            num_classes = 47

        M = 1

        # Some log files for metrics and debug info
        logging.write_intro_logs(["server_evaluation"], setting, algorithm)

        server_model = create_cnn_model(False)

        server_model.compile(
            optimizer=keras.optimizers.SGD(0.1),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )
        if algorithm == "fedgkd":
            historical_ensemble_model = create_cnn_model(False)
            historical_ensemble_model.compile(
                optimizer=keras.optimizers.SGD(0.1),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
            )
            historical_ensemble_model.set_weights(server_model.get_weights())
            historical_models = []


        if dataset == "emnist":
            emnist_tfds = tfds.load("emnist/balanced")
            test_ds = emnist_tfds["test"]
            test_ds = test_ds.map(lambda element: (tf.cast(element["image"], tf.float32) / 255.0, element["label"]))\
                .batch(test_batch_size)

        np.random.seed(random_seed)
        # tensorboard
        # logs/dataset_alpha/algorithm/specific_parameters,seed
        logdir = os.path.join(PATH, "logs", dataset + "_" + str(round(alpha, 2)), algorithm,
                              logging.encode_hyperparameter_in_string(setting))
        global_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "global_test"))

        history = server_model.evaluate(test_ds, return_dict=True)
        with global_summary_writer.as_default():
            tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=0)
            tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=0)

        for rnd in range(1, total_rounds + 1):
            # cifar10 is partitioned among 20 clients
            list_of_clients = [i for i in range(0, total_clients)]
            sampled_clients = np.random.choice(
                list_of_clients,
                # [10, 10, 10, 10, 10],
                size=total_clients,
                replace=False)
            print("Selected clients for the next round ", sampled_clients)

            mean_client_loss = 0
            mean_client_accuracy = 0

            global_weights = server_model.get_weights()
            aggregated_weights = tf.nest.map_structure(lambda a: a,
                                                       global_weights)

            global_weights = tf.nest.map_structure(lambda a, b: a - b,
                                                   global_weights,
                                                   global_weights)

            selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)
            print("total examples", total_examples)

            for c in range(0, total_clients):
                print("Client: ", c)
                local_examples = selected_client_examples[c]
                print("Local examples: ", local_examples)

                # Local training
                print(f"[Client {c}] Local Training ---")

                training_dataset = data_utility.load_client_datasets_from_files(
                    dataset=dataset,
                    sampled_client=sampled_clients[c],
                    batch_size=local_batch_size,
                    alpha=alpha)

                cnn_simple_model = create_cnn_model(False)

                cnn_simple_model.set_weights(aggregated_weights)
                if algorithm == "fedavg":
                    client_model = cnn_simple_model
                    client_model.compile(keras.optimizers.SGD(0.1),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         )

                if algorithm == "fedprox":
                    mu = setting["mu"]
                    client_model = prox.FedProxModel(cnn_simple_model)
                    client_model.compile(keras.optimizers.SGD(0.1),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         initial_weights=aggregated_weights,
                                         mu=mu)

                if algorithm == "fedgkd":
                    gamma = setting["gamma"]
                    client_model = gkt.FedGKDModel(cnn_simple_model)
                    client_model.compile(keras.optimizers.SGD(0.1),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(),
                                         gamma=gamma)

                # client_model.set_weights(aggregated_weights)

                if algorithm == "fedgkd":
                    teacher_logits = historical_ensemble_model.predict(training_dataset)
                    teacher_logits_ds = tf.data.Dataset.from_tensor_slices(teacher_logits).batch(local_batch_size)
                    training_dataset = tf.data.Dataset.zip((training_dataset,
                                                            teacher_logits_ds))


                history = client_model.fit(
                    training_dataset,
                    batch_size=local_batch_size,
                    epochs=local_epochs,
                )

                global_weights = tf.nest.map_structure(lambda a, b: a + (local_examples / total_examples) * b,
                                                       global_weights,
                                                       client_model.model.get_weights() if algorithm not in ["fedavg",
                                                                                                             "fedprox"] else
                                                       client_model.get_weights())

                # print(global_weights[0])
                loss, accuracy = client_model.evaluate(test_ds)

                mean_client_loss = mean_client_loss + loss / total_clients
                mean_client_accuracy = mean_client_accuracy + accuracy / total_clients

            # Server evaluation
            server_model.set_weights(global_weights)

            if algorithm == "fedgkd":
                if len(historical_models) == M:
                    historical_models.pop(0)
                historical_models.append(global_weights)

                # Compute ensemble of M historical global models
                # Initialize the ensemble to 0
                historical_ensemble_weights = tf.nest.map_structure(lambda a, b: a - b,
                                                                    historical_models[0],
                                                                    historical_models[0])
                for model_weights in historical_models:
                    historical_ensemble_weights = tf.nest.map_structure(lambda a, b: a + b / len(historical_models),
                                                                        historical_ensemble_weights,
                                                                        model_weights)

                historical_ensemble_model.set_weights(historical_ensemble_weights)

            print(f"[info] mean client loss {mean_client_loss}")
            print(f"[info] mean client accuracy {mean_client_accuracy}")

            print("[Server] Evaluation - Round: ", rnd)
            # history = client_model.evaluate(test_ds, return_dict=True)
            history = server_model.evaluate(test_ds, return_dict=True)
            with global_summary_writer.as_default():
                tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)
            logging.write_logs("server_evaluation", history, algorithm)
