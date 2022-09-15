"""This is a minimal implementation of Federated Distillation based on the papers:
https://arxiv.org/abs/2008.06180
https://ieeexplore.ieee.org/abstract/document/9435947
"""
# tensorboard dev upload --logdir /home/amora/pycharm_projects/fed_dist_simulated/exp_results/logs --name "Cifar10 8 clients" --description "Comparison of algorithms."
# https://tensorboard.dev/experiment/ULzAiK7SQWCE4LCox9tnUA/#scalars&regexInput=0.1%2Fglo
import os

# This is for explicitly use GPU recognized as device 0
# Remove this line if you dont need it
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import data_utility
import tensorflow as tf
import numpy as np
from tensorflow import keras
import residual_model as res
import itertools as it
import FedGKDModel as gkt
import FedProxModel as prox
import tensorflow_datasets as tfds

# import datetime

PATH = "exp_results_batch_norm_v2"

def get_all_hp_combinations(hp):
    '''Turns a dict of lists into a list of dicts'''
    combinations = it.product(*(hp[name] for name in hp))
    hp_dicts = [{key: value[i] for i, key in enumerate(hp)} for value in combinations]
    return hp_dicts


# Just some utilities for log purposes
def write_logs(file_name, history, folder, client=None):
    folder = os.path.join(PATH, folder)
    prefix = ""
    if client is not None:
        prefix = "[" + str(client) + "]" + " "

    f = open(os.path.join(folder, file_name), "a")
    f.write(prefix + str(history))
    f.write("\n")
    f.close()
    return


def write_intro_logs(list_file_names, dict_sett, folder):
    folder = os.path.join(PATH, folder)

    exist = os.path.exists(folder)

    if not exist:
        os.makedirs(folder)

    for file_name in list_file_names:
        f = open(os.path.join(folder, file_name), "a")
        f.write("Simulation Start " + str(dict_sett))
        f.write("\n")
        f.close()
    return


def zip_img_and_pred(ds, teacher_pred, batch_size_distill, shuffle_size=1024):
    """Zip the input data with teacher predictions. Also shuffle and batch for training."""
    img_ds = ds.map(lambda image, label: image)
    zipped_img_pred = tf.data.Dataset.zip((img_ds, teacher_pred))
    zipped_img_pred = zipped_img_pred.shuffle(shuffle_size)
    zipped_img_pred = zipped_img_pred.batch(
        batch_size_distill)
    return zipped_img_pred


def element_norm_fn(image, label):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.4914, 0.4822, 0.4465],
                                               variance=[np.square(0.2023),
                                                         np.square(0.1994),
                                                         np.square(0.2010)])
    return norm_layer(tf.cast(image, tf.float32) / 255.0), label

def element_norm_fn_cifar100(element):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.4914, 0.4822, 0.4465],
                                               variance=[np.square(0.2023),
                                                         np.square(0.1994),
                                                         np.square(0.2010)])
    return norm_layer(tf.cast(element["image"], tf.float32) / 255.0), element["label"]


def reinitialize_classifier(model):
    found = False
    for layer in model.layers:
        # print(layer.name)
        if layer.name == 'classifier':
            # print(layer.get_weights())
            layer.set_weights([layer.kernel_initializer(shape=layer.kernel.shape),
                               layer.bias_initializer(shape=layer.bias.shape)])
            # print(layer.get_weights())
            found = True
    if not found:
        print("Classifier layer not found!")
    return found


def encode_hyperparameter_in_string(s):
    hyperparameters = ["E", "C", "gamma", "seed"]
    if s["algorithm"] == "fedavg":
        hyperparameters = ["E", "C", "seed"]
        
    encoded = ""

    for h in hyperparameters:
        if h =="seed":
            encoded = encoded + "," + h + str(round(s[h], 3))
        else:
            if encoded == "":
                encoded = encoded + h + str(round(s[h], 3))
            else:
                encoded = encoded + "_" + h + str(round(s[h], 3))
    return encoded


if __name__ == '__main__':

    # Building a dictionary of hyperparameters
    hp = {}
    # hp["algorithm"] = ["fedavg", "fedprox"]
    hp["algorithm"] = ["fedprox"]
    # hp["algorithm"] = ["feddistill_plus"]
    # hp["algorithm"] = ["feddistill"]
    hp["reinit_classifier"] = [False]
    hp["batch_size"] = [32]
    hp["E"] = [10] # local_epochs
    hp["C"] = [8] # n clients
    hp["rounds"] = [100]
    hp["alpha"] = [100.0]

    hp["norm"] = ["batch"]
    hp["dataset"] = ["cifar10"]

    hp["gamma"] = [0.0001]
    hp["t"] = [0.1] #temperature
    hp["M"] = [1] # only for fedgdk
    hp["seed"] = [2021, 2020] # seed for client selection

    # Creating a list of dictionaries
    # each one for a combination of hp
    settings = get_all_hp_combinations(hp)

    # Running a simulation for each of the setting in hp possible combinations
    for setting in settings:
        print("simulation start " + str(setting))
        total_rounds = setting["rounds"]
        total_clients = setting["C"]
        local_epochs = setting["E"] # local_epochs
        local_batch_size = setting["batch_size"]
        test_batch_size = 256
        reinit_classifier = setting["reinit_classifier"]
        alpha = setting["alpha"]
        M = setting["M"]
        algorithm = setting["algorithm"]
        norm = setting["norm"]
        load_stl10_from_file = False
        random_seed = setting["seed"]
        dataset = setting["dataset"]
        temperature = setting["t"]

        if algorithm == "fedprox":
            mu = setting["gamma"]
        else:
            gamma = setting["gamma"]

        num_classes = 10
        if dataset == "cifar100":
            num_classes = 100

        # Some log files for metrics and debug info
        write_intro_logs(["server_evaluation", "client_local_training_history",
                          "average_client_evaluation_after_local_training"], setting, algorithm)

        server_model = res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm)

        server_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )
        if algorithm == "fedgkd":
            historical_ensemble_model = res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm)
            historical_ensemble_model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
            )
            historical_ensemble_model.set_weights(server_model.get_weights())
            historical_models = []

        if dataset == "cifar10":
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(element_norm_fn).batch(
                test_batch_size)

        if dataset == "cifar100":
            cifar100_tfds = tfds.load("cifar100")
            test_ds = cifar100_tfds["test"]
            test_ds = test_ds.map(element_norm_fn_cifar100).batch(test_batch_size)

        np.random.seed(random_seed)
        # tensorboard
        # logs/dataset_alpha/algorithm/specific_parameters
        # C8_E10_batch_lambda0.1_temperature1.0 _M1
        logdir = os.path.join(PATH, "logs", dataset + "_" + str(round(alpha, 2)), algorithm, encode_hyperparameter_in_string(setting))
        local_train_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "local_train"))
        local_test_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "local_test"))
        global_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "global_test"))

        history = server_model.evaluate(test_ds, return_dict=True)
        with global_summary_writer.as_default():
            tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=0)
            tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=0)

        for rnd in range(1, total_rounds + 1):
            # cifar10 is partitioned among 20 clients
            list_of_clients = [i for i in range(0, 20)]
            sampled_clients = np.random.choice(
                list_of_clients,
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

            if algorithm == "feddistill_plus" or algorithm == "feddistill":
                global_mean_per_label_soft_labels = tf.zeros([10, 10], tf.float32)

            selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)

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

                resnet18_model = res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm)
                if algorithm != "feddistill": # fed_distill does not exchange weights
                    resnet18_model.set_weights(aggregated_weights)

                if algorithm == "fedavg":
                    client_model = resnet18_model
                    client_model.compile(optimizer=keras.optimizers.Adam(),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         )

                if algorithm == "fedprox":
                    client_model = prox.FedProxModel(resnet18_model)
                    client_model.compile(optimizer=keras.optimizers.Adam(),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         initial_weights=aggregated_weights,
                                         mu=mu)

                if algorithm == "fedgkd":
                    client_model = gkt.FedGKDModel(resnet18_model)
                    client_model.compile(optimizer=keras.optimizers.Adam(),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence())

                if algorithm == "fedntd":
                    client_model = ntd.FedNTDModel(resnet18_model)
                    client_model.compile(optimizer=keras.optimizers.Adam(),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence())

                if algorithm == "feddistill_plus" or algorithm == "feddistill":
                    client_model = distill_plus.FedDistillPlusModel(resnet18_model)
                    client_model.compile(optimizer=keras.optimizers.Adam(),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(),
                                         gamma=0.0 if rnd == 1 else 0.1,
                                         global_mean_soft_labels=global_mean_per_label_soft_labels)

                # client_model.set_weights(aggregated_weights)

                if algorithm == "fedgkd":
                    teacher_logits = historical_ensemble_model.predict(training_dataset)
                    teacher_logits_ds = tf.data.Dataset.from_tensor_slices(teacher_logits).batch(local_batch_size)
                    training_dataset = tf.data.Dataset.zip((training_dataset,
                                                            teacher_logits_ds))

                if algorithm == "fedntd":
                    teacher_logits = server_model.predict(training_dataset)
                    teacher_logits_ds = tf.data.Dataset.from_tensor_slices(teacher_logits).batch(local_batch_size)
                    training_dataset = tf.data.Dataset.zip((training_dataset,
                                                            teacher_logits_ds))

                history = client_model.fit(
                    training_dataset,
                    batch_size=local_batch_size,
                    epochs=local_epochs,
                )
                # log on tensorboard
                # print(history.history["loss"])
                # with local_train_summary_writer.as_default():
                #     tf.summary.scalar('loss', history.history["loss"][len(history.history["loss"])-1], step=rnd)
                #     tf.summary.scalar('accuracy', history.history["accuracy"][len(history.history["loss"])-1], step=rnd)

                write_logs("client_local_training_history", history.history, algorithm, client=c)

                if algorithm != "feddistill":
                    global_weights = tf.nest.map_structure(lambda a, b: a + (local_examples*b) / total_examples,
                                                           global_weights,
                                                           client_model.model.get_weights() if algorithm not in ["fedavg", "fedprox"] else
                                                           client_model.get_weights())

                # print(global_weights[0])
                loss, accuracy = client_model.evaluate(test_ds)
                with local_test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=rnd)
                    tf.summary.scalar('accuracy', accuracy, step=rnd)

                mean_client_loss = mean_client_loss + loss / total_clients
                mean_client_accuracy = mean_client_accuracy + accuracy / total_clients

                if algorithm == "feddistill_plus" or algorithm == "feddistill":
                    # print("label_count ", client_model.get_label_count())
                    # print("per_label_soft_labels ", client_model.get_per_label_soft_label())
                    local_mean_per_label_soft_labels = tf.math.divide_no_nan(client_model.get_per_label_soft_label(),
                                                                             tf.transpose(tf.expand_dims(
                                                                                 client_model.get_label_count(),
                                                                                 axis=0))
                                                                             )
                    # print("mean_per_label_soft_labels", local_mean_per_label_soft_labels)
                    global_mean_per_label_soft_labels = global_mean_per_label_soft_labels + \
                                                        local_mean_per_label_soft_labels / total_clients

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
            write_logs("average_client_evaluation_after_local_training",
                       "accuracy " + str(mean_client_accuracy) + " loss " + str(mean_client_loss), algorithm)

            print("[Server] Evaluation - Round: ", rnd)
            history = server_model.evaluate(test_ds, return_dict=True)
            with global_summary_writer.as_default():
                tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)
            write_logs("server_evaluation", history, algorithm)
