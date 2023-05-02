import os
import shutil
# This is for explicitly use GPU recognized as device 0
# Remove this line if you dont need it
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import data_utility
import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow import keras
# import residual_model as res
import fedfa_utility
import residual_model_custom as res
from non_iid_algorithms.FedDynModel import FedDynModel
from non_iid_algorithms.FedGKDModel import FedGKDModel
from non_iid_algorithms.FedNTDModel import FedNTDModel
from non_iid_algorithms.FedProxModel import FedProxModel
from non_iid_algorithms.FedLGICModel import FedLGICModel
from non_iid_algorithms.FedLGICDModel import FedLGICDModel
from non_iid_algorithms.FedMLB2Model import FedMLB2Model
from non_iid_algorithms.FedLCModel import FedLCModel
from non_iid_algorithms.MoonModel import MoonModel
from non_iid_algorithms.FedDynMLBModel import FedDynMLBModel
from non_iid_algorithms.FedMAXModel import FedMAXModel
from training_loop_custom import train_one_epoch

import tensorflow_datasets as tfds
import logging_utility as logging
from typing import Optional
import gc

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(
#         device=gpu, enable=True
#     )
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

PATH = os.path.join("determinism_3")


# ---------- clear memory -----------------
def reset_tensorflow_keras_backend():
    # to be further investigated, but this seems to be enough
    tf.keras.backend.clear_session()
    # tf.reset_default_graph()
    _ = gc.collect()
    return

def zip_img_and_pred(ds, teacher_pred, batch_size_distill, shuffle_size=1024):
    """Zip the input data with teacher predictions. Also shuffle and batch for training."""
    img_ds = ds.map(lambda image, label: image)
    zipped_img_pred = tf.data.Dataset.zip((img_ds, teacher_pred))
    zipped_img_pred = zipped_img_pred.shuffle(shuffle_size)
    zipped_img_pred = zipped_img_pred.batch(
        batch_size_distill)
    return zipped_img_pred


def element_norm_fn_cifar10(image, label):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.4914, 0.4822, 0.4465],
                                               variance=[np.square(0.2470),
                                                         np.square(0.2435),
                                                         np.square(0.2616)])
    return norm_layer(tf.cast(image, tf.float32) / 255.0), label


# cifar10
# mean
# tf.Tensor([0.49139968 0.48215841 0.44653091], shape=(3,), dtype=float64)
# std
# tf.Tensor([0.24703223 0.24348513 0.26158784], shape=(3,), dtype=float64)
# cifar100
# mean
# tf.Tensor([0.50707516 0.48654887 0.44091784], shape=(3,), dtype=float64)
# std
# tf.Tensor([0.26733429 0.25643846 0.27615047], shape=(3,), dtype=float64)
def element_norm_fn_cifar100(element):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.5071, 0.4865, 0.4409],
                                               variance=[np.square(0.2673),
                                                         np.square(0.2564),
                                                         np.square(0.2762)])
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


def create_model(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                   l2_weight_decay=l2_weight_decay, seed=seed)
    else:
        return res.create_resnet8(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                  l2_weight_decay=l2_weight_decay, seed=seed)


def create_model_mlb(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18_mlb(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                       l2_weight_decay=l2_weight_decay, seed=seed)
    else:
        return res.create_resnet8_mlb(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                      l2_weight_decay=l2_weight_decay, seed=seed)


def create_model_fa(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18_fa(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                      l2_weight_decay=l2_weight_decay, seed=seed)


def create_model_max(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18_max(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                       l2_weight_decay=l2_weight_decay, seed=seed)


def create_model_moon(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18_moon(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                        l2_weight_decay=l2_weight_decay, seed=seed)
    else:
        # Not implemented
        return None


def create_model_lgic(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18_lgic(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                        l2_weight_decay=l2_weight_decay, seed=seed)
    else:
        return res.create_resnet8_lgic(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                       l2_weight_decay=l2_weight_decay, seed=seed)


def create_server_optimizer(optimizer="sgd", momentum=0.0, lr=1.0):
    if optimizer == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    elif optimizer == "adam":
        return keras.optimizers.Adam(learning_rate=lr, epsilon=1e-3)


def load_local_param_curr_from_file(client_id, c_list, server_model, algorithm="feddyn", architecture="resnet18",
                                    num_classes=100, norm="group", random_seed=None):
    if algorithm in ["feddyn", "feddynmlb"]:
        model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                             l2_weight_decay=0.0)
    else:
        model = create_model_moon(architecture=architecture, num_classes=num_classes, norm=norm,
                                  l2_weight_decay=0.0, seed=random_seed)
    path = os.path.join(algorithm + "_checkpoints", "client" + str(client_id), "saved_local_param")
    if c_list[client_id] != 0:
        model.load_weights(path)
        to_return_weights = model.get_weights()
    else:
        to_return_weights = tf.nest.map_structure(lambda a, b: a - b,
                                                  server_model.get_weights(),
                                                  server_model.get_weights()) if algorithm == "feddyn" else model.get_weights()
    return to_return_weights


def save_local_param_to_file(client_id, local_param, c_list, algorithm="feddyn", architecture="resnet18",
                             num_classes=100, norm="group"):
    # saving
    if algorithm in ["feddyn", "feddynmlb"]:
        model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                             l2_weight_decay=0.0)
    else:
        model = create_model_moon(architecture=architecture, num_classes=num_classes, norm=norm,
                                  l2_weight_decay=0.0)
    model.set_weights(local_param)

    path = os.path.join(algorithm + "_checkpoints", "client" + str(client_id), "saved_local_param")
    model.save_weights(path)

    c_list[client_id] = 1
    return


# def exp_decayed_learning_rate(initial_learning_rate, decay_rate, step, decay_steps):
#     return initial_learning_rate * decay_rate ** (step / decay_steps)


if __name__ == '__main__':
    # Building a dictionary of hyperparameters
    hp = {}
    # hp["algorithm"] = ["feddynmlb"]
    hp["algorithm"] = ["fedlc"]
    # hp["algorithm"] = ["moon"]
    hp["reinit_classifier"] = [False]
    hp["updates"] = [50]
    hp["E"] = [5]  # local_epochs
    hp["C"] = [5]  # n clients
    hp["total_clients"] = [100]
    hp["rounds"] = [500]
    hp["alpha"] = [0.3]
    hp["norm"] = ["group"]
    hp["dataset"] = ["cifar100"]
    hp["lr_client"] = [0.1]
    hp["momentum"] = [0.0]
    hp["weight_decay"] = [1e-3]
    hp["architecture"] = ["resnet18"]
    hp["server_side_optimizer"] = ["sgd"]
    hp["lr_server"] = [1.0]
    hp["server_momentum"] = [0.0]
    hp["lr_decay"] = [0.998]
    hp["clipnorm"] = [10.0]
    hp["augm_"] = ["y"]
    hp["part"] = ["AVG5"]

    hp["seed"] = [0]  # seed for client selection, model initialization, client data shuffling

    # algorithm-specific hyperparameter
    cd = {}
    # cd["fedgkd"] = {"M": [5], "gamma": [0.2]}
    cd["fedmax"] = {"beta": [0.1]}  # 1000, 15000
    # cd["fedsam"] = {"rho": [0.05], "part": [2]}
    # cd["fedlgic"] = {"lambda1_lambda2_": [[0.7, 0.7], [1.0, 1.0]]}
    # cd["fedlgicd"] = {"lambda1_lambda2_": [[0.1, 0.1], [0.15, 0.15]]}
    # cd["fedprox"] = {"mu": [0.01, 0.001]}
    # cd["feddyn"] = {"alpha_dyn": [0.1]}
    # cd["feddynmlb"] = {"alpha_dyn": [0.1], "lambda1_lambda2_": [[1.0, 1.0]]}
    # cd["moon"] = {"mu_moon": [1.0], "t_moon": [0.5]}
    # cd["fedntd"] = {"beta": [1.0, 0.3]}
    # cd["fedmlb"] = {"lambda1_lambda2_": [[1.0, 1.0]]}
    # cd["feddlb"] = {"alpha_dlb": [0.1, 0.2, 0.3, 0.5, 1.0, 0.01, 3.0]}
    # cd["feddlb"] = {"alpha_dlb": [0.0], "temperature":[1.0]}
    cd["fedlc"] = {"tau": [1.0]}
    # Creating a list of dictionaries
    # each one for a combination of hp + algorithm-specific hyperparams
    settings = logging.get_combinations(hp, cd)

    # Running a simulation for each of the setting in hp possible combinations
    for setting in settings:
        print("Simulation start with configuration " + str(setting))
        total_rounds = setting["rounds"]
        k = setting["C"]
        total_clients = setting["total_clients"]
        local_epochs = setting["E"]  # local_epochs
        # local_batch_size = setting["batch_size"]
        local_updates = setting["updates"]
        test_batch_size = 256
        reinit_classifier = setting["reinit_classifier"]
        alpha = setting["alpha"]
        algorithm = setting["algorithm"]
        norm = setting["norm"]  # group, batch, layer norm
        load_stl10_from_file = False
        random_seed = setting["seed"]
        dataset = setting["dataset"]
        lr_client_initial = setting["lr_client"]
        l2_weight_decay = setting["weight_decay"]
        momentum = setting["momentum"]
        architecture = setting["architecture"]
        lr_server = setting["lr_server"]
        server_side_optimizer = setting["server_side_optimizer"]
        server_momentum = setting["server_momentum"]
        exp_decay = setting["lr_decay"]
        num_classes = 10
        clipnorm = 10.0  # as FedMLB
        tf.keras.utils.set_random_seed(random_seed)
        # tf.config.experimental.enable_op_determinism()
        continuation = False
        interrupting_round = 500
        reporting_exec_time_enabled = False
        if continuation:
            restarting_round = interrupting_round
        else:
            restarting_round = 0

        reporting_rounds = range(350, 500)
        exec_time_mean_reporting_rounds = 0.0
        local_test = False

        if dataset == "cifar100":
            num_classes = 100

        # Some log files for metrics and debug info
        # logging.write_intro_logs(["server_evaluation", "client_local_training_history",
        #                           "average_client_evaluation_after_local_training"], setting, algorithm)

        lr_client = lr_client_initial
        if algorithm != "moon":
            server_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm, seed=random_seed)
        else:
            server_model = create_model_moon(architecture=architecture, num_classes=num_classes, norm=norm,
                                             seed=random_seed)

        server_optimizer = create_server_optimizer(optimizer=server_side_optimizer, lr=lr_server,
                                                   momentum=server_momentum)

        server_model.compile(
            optimizer=server_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )
        server_model.summary(expand_nested=True)
        if algorithm == "fedgkd":
            historical_ensemble_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                     seed=random_seed)
            historical_ensemble_model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=lr_server, momentum=server_momentum),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
            )
            historical_ensemble_model.set_weights(server_model.get_weights())
            historical_weights = []
            M = setting["M"]

        if algorithm in ["feddyn", "feddynmlb"]:
            all_clients_model = False
            client_list = [0 for _ in range(total_clients)]

            weight_list = np.asarray([500.0 for i in range(100)])
            if alpha == 0.1:
                weight_list = data_utility.load_all_clients_statistics(alpha, dataset)

            weight_list = weight_list / np.sum(weight_list) * total_clients

            cld_mdl_param = tf.nest.map_structure(lambda a: a,
                                                  server_model.get_weights(),
                                                  )
            # init
            if all_clients_model:
                all_clnt_avg = tf.nest.map_structure(lambda a: a,
                                                     server_model.get_weights()
                                                     )

            local_param_avg = tf.nest.map_structure(lambda a, b: a - b,
                                                    server_model.get_weights(),
                                                    server_model.get_weights())
            if continuation:
                path = os.path.join(algorithm + "_checkpoints", "local_param_avg_model", "local_param_avg")
                server_model.load_weights(path)
                local_param_avg = server_model.get_weights()

                path = os.path.join(algorithm + "_checkpoints", "server_model", "saved_local_param")
                server_model.load_weights(path)
                cld_mdl_param = tf.nest.map_structure(lambda a: a,
                                                      server_model.get_weights(),
                                                      )
                lr_client = lr_client * (exp_decay ** interrupting_round)
                client_list = [1 for _ in range(total_clients)]

            if not continuation:
                # delete already created checkpoints
                folder_path = algorithm + "_checkpoints"
                exist = os.path.exists(folder_path)
                if exist:
                    shutil.rmtree(folder_path, ignore_errors=True)

        if algorithm in ["fedsam", "fedfa", "feddlb"]:
            if continuation:
                path = os.path.join(algorithm + "_checkpoints", "server_model", "saved_local_param")
                server_model.load_weights(path)
                # server_model.load_model(path)
                path = os.path.join(algorithm + "_checkpoints", "opt_dict.pickle")


                def dic_load(filename):
                    # with open(filename + '.json') as fp:
                    #     return json.load(fp)
                    with open(filename, 'rb') as fp:
                        return pickle.load(fp)


                server_model.optimizer.from_config(dic_load(path))
                if algorithm == "fedfa":
                    path = os.path.join(algorithm + "_checkpoints", "mean_std.pickle")
                    fedfa_dict = dic_load(path)
                    aggregated_non_train_global_mean = fedfa_dict["aggregated_mean"]
                    aggregated_non_train_global_std = fedfa_dict["aggregated_std"]

                # server_model = tf.saved_model.load(path)
                # ww = server_model.get_weights()
                # print("ww OK!")
                # path2 = os.path.join(algorithm + "_checkpoints", "server_model", "saved_local_param2")
                # server_model.save.save(path2, format="h5")
                # server_model = tf.saved_model.load(path)
                lr_client = lr_client * (exp_decay ** interrupting_round)

            if not continuation:
                # delete already created checkpoints
                folder_path = algorithm + "_checkpoints"
                exist = os.path.exists(folder_path)
                if exist:
                    shutil.rmtree(folder_path, ignore_errors=True)

        if algorithm == "moon":
            client_list = [0 for _ in range(total_clients)]
            folder_path = "moon_checkpoints"
            exist = os.path.exists(folder_path)
            if exist:
                shutil.rmtree(folder_path, ignore_errors=True)

        if dataset == "cifar10":
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(element_norm_fn_cifar10).batch(
                test_batch_size)

        if dataset == "cifar100":
            cifar100_tfds = tfds.load("cifar100")
            test_ds = cifar100_tfds["test"]
            test_ds = test_ds.map(element_norm_fn_cifar100).batch(test_batch_size)

        # tensorboard
        # ..../dataset_alpha--_C--_k--/algorithm/general_hyperparameters/specific_hyperparameters,seed---
        logdir = os.path.join(PATH, dataset + "_alpha" + str(round(alpha, 2)) + "_C" + str(round(total_clients, 2)) +
                              "_k" + str(round(k, 2)) + "_E" + str(round(local_epochs, 2)), architecture,
                              logging.encode_general_hyperparameter_in_string(setting),
                              algorithm,
                              logging.encode_specific_hyperparameter_in_string(setting) + "," + "seed" + str(
                                  round(random_seed, 2)))
        if local_test:
            local_test_lda_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "local_test_lda"))
            local_test_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "local_test"))

        global_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "global_test"))
        if algorithm in ["feddyn", "feddynmlb"] and all_clients_model:
            global_dyn_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "all_test"))

        # historical_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "historical_test"))

        history = server_model.evaluate(test_ds, return_dict=True)
        with global_summary_writer.as_default():
            tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=0)
            tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=0)

        for rnd in range(restarting_round + 1, total_rounds + 1):
            # for rnd in range(1, total_rounds + 1):
            # cifar10 is partitioned among 20 clients
            # if alpha == 0.3:
            #     if rnd > 10:
            #         if tf.squeeze(history["accuracy"]) < (1.0/num_classes)*2.5: # 0.03 for cifar100
            #             # print("CONTINUE")
            #             continue
            #
            #     if rnd > 200:
            #         # if tf.squeeze(history["accuracy"]) < 0.34:
            #         #     continue
            #         continue

            list_of_clients = [i for i in range(0, total_clients)]
            sampled_clients = np.random.choice(
                list_of_clients,
                size=k,
                replace=False)
            print("Selected clients for the next round ", sampled_clients)

            mean_client_loss = 0
            mean_client_accuracy = 0
            mean_client_loss_lda = 0
            mean_client_accuracy_lda = 0

            if algorithm == "fedfa":
                fedfa_non_train_global_std = []
                fedfa_non_train_global_mean = []

            # if algorithm != "feddyn" or (algorithm == "feddyn" and rnd == 1):
            if algorithm not in ["feddyn", "feddynmlb"]:
                aggregated_weights = tf.nest.map_structure(lambda a: a,
                                                           server_model.get_weights())

                new_w_global = tf.nest.map_structure(lambda a, b: a - b,
                                                     server_model.get_weights(),
                                                     server_model.get_weights())

                # global_weights = tf.nest.map_structure(lambda a, b: a - b,
                #                                        global_weights,
                #                                        global_weights)

                delta_w_global_trainable = tf.nest.map_structure(lambda a, b: a - b,
                                                                 server_model.trainable_weights,
                                                                 server_model.trainable_weights)
                if norm in ["batch"]:
                    delta_w_with_batch = tf.nest.map_structure(lambda a, b: a - b,
                                                               server_model.get_weights(),
                                                               server_model.get_weights())

            else:
                # if rnd == 501:
                #     path = os.path.join(algorithm + "_checkpoints", "local_param_avg_model", "local_param_avg")
                #     server_model.load_weights(path)
                #     local_param_avg = server_model.get_weights()
                #
                #     path = os.path.join(algorithm + "_checkpoints", "server_model", "saved_local_param")
                #     server_model.load_weights(path)
                #     cld_mdl_param = tf.nest.map_structure(lambda a: a,
                #                                           server_model.get_weights(),
                #                                           )
                # init
                if all_clients_model:
                    all_clnt_avg = tf.nest.map_structure(lambda a: a * ((total_clients - k) / total_clients),
                                                         all_clnt_avg
                                                         )

                active_clnt_avg = tf.nest.map_structure(lambda a, b: a - b,
                                                        server_model.get_weights(),
                                                        server_model.get_weights())

                # this is not equal to the original strategy
                # local_param_avg = tf.nest.map_structure(lambda a: a * ((total_clients - k) / total_clients),
                #                                      local_param_avg
                #                                      )

            #
            # w_init_non_trainable = tf.nest.map_structure(lambda a, b: a - b,
            #                                      server_model.non_trainable_weights,
            #                                      server_model.non_trainable_weights)

            if algorithm == "feddistill_plus" or algorithm == "feddistill":
                global_mean_per_label_soft_labels = tf.zeros([num_classes, num_classes], tf.float32)

            selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)

            exec_time_mean_round = 0.0

            for c in range(0, k):
                # release keras resource
                # tf.keras.backend.clear_session()
                print("Client: ", c)
                local_examples = selected_client_examples[c]
                print("Local examples: ", local_examples)

                # Local training
                print(f"[Client {c}] Local Training ---")

                # setting the local_batch_size to perform a total of local_updates
                local_batch_size = round(local_examples * local_epochs / local_updates)

                training_dataset = data_utility.load_client_datasets_from_files(
                    dataset=dataset,
                    sampled_client=sampled_clients[c],
                    batch_size=local_batch_size,
                    alpha=alpha,
                    seed=random_seed)

                if algorithm not in ["feddyn", "feddynmlb", "moon", "fedavg", "fedfa", "fedsam"]:
                    resnet_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                l2_weight_decay=l2_weight_decay, seed=random_seed)

                if algorithm not in ["feddistill", "feddyn", "feddynmlb",
                                     "moon", "fedsam", "fedfa", "fedavg",
                                     "fedmlb", "fedmax"]:  # fed_distill does not exchange weights
                    resnet_model.set_weights(aggregated_weights)

                if algorithm in ["fedavg", "feddlb"]:
                    client_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                l2_weight_decay=0.0, seed=random_seed)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm,
                                                                        weight_decay=l2_weight_decay),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         )
                    client_model.set_weights(server_model.get_weights())
                    # if norm not in ["batch"]:
                    #     client_model.set_weights(aggregated_weights)
                    # else:
                    #     client_model._non_trainable_weights = tf.nest.map_structure(lambda a: a,
                    #                                                                 server_model.non_trainable_weights)
                    #     client_model._trainable_weights = tf.nest.map_structure(lambda a: a,
                    #                                                             server_model.trainable_weights)

                if algorithm in ["fedfa"]:
                    client_model = create_model_fa(architecture=architecture, num_classes=num_classes, norm=norm,
                                                   l2_weight_decay=0.0, seed=random_seed)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm,
                                                                        weight_decay=l2_weight_decay),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         )
                    tf.nest.map_structure(lambda a, b: a.assign(b),
                                          client_model.trainable_variables,
                                          server_model.get_weights())

                    if rnd > 1:
                        client_model.set_running_var(var_mean=aggregated_non_train_global_mean,
                                                     var_std=aggregated_non_train_global_std)

                if algorithm in ["fedmax"]:
                    beta = setting["beta"]
                    resnet_model = create_model_max(architecture=architecture, num_classes=num_classes, norm=norm,
                                                    l2_weight_decay=0.0, seed=random_seed)
                    client_model = FedMAXModel(resnet_model)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm,
                                                                        weight_decay=l2_weight_decay),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(),
                                         beta=beta
                                         )
                    client_model.set_weights(server_model.get_weights())

                if algorithm in ["fedlc"]:
                    class_occ = data_utility.load_class_occurrence(sampled_clients, alpha, dataset)
                    tau = setting["tau"]
                    # print(prob)
                    # print(tf.reduce_sum(prob, axis=1))
                    resnet_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                    l2_weight_decay=0.0, seed=random_seed)
                    client_model = FedLCModel(resnet_model)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm,
                                                                        weight_decay=l2_weight_decay),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         class_occ=class_occ[c],
                                         tau=tau
                                         )
                    client_model.set_weights(server_model.get_weights())

                if algorithm == "fedsam":
                    rho = setting["rho"]
                    resnet_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                l2_weight_decay=0.0,
                                                seed=random_seed)
                    resnet_model.set_weights(aggregated_weights)
                    client_model = tf.keras.models.experimental.SharpnessAwareMinimization(resnet_model, rho=rho)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm,
                                                                        weight_decay=l2_weight_decay),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         )

                if algorithm == "fedprox":
                    mu = setting["mu"]
                    client_model = FedProxModel(resnet_model)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         initial_weights=aggregated_weights,
                                         mu=mu)
                    # client_model.set_weights(server_model.get_weights())

                if algorithm == "feddyn":
                    alpha_dyn = setting["alpha_dyn"]
                    alpha_coef_adpt = alpha_dyn / weight_list[sampled_clients[c]]  # adaptive alpha coef

                    resnet_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                # l2_weight_decay=(alpha_coef_adpt + l2_weight_decay),
                                                l2_weight_decay=0.0,
                                                seed=random_seed)
                    resnet_model.set_weights(server_model.get_weights())

                    local_param_list_curr = load_local_param_curr_from_file(sampled_clients[c], client_list,
                                                                            server_model)

                    client_model = FedDynModel(resnet_model)
                    opt = tf.keras.optimizers.experimental.SGD(learning_rate=lr_client, momentum=momentum,
                                                               clipnorm=clipnorm,
                                                               weight_decay=(alpha_coef_adpt + l2_weight_decay))
                    # opt = keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum, clipnorm=clipnorm)
                    client_model.compile(optimizer=opt,
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         avg_mdl_param=server_model.get_weights(),
                                         local_grad_vector=local_param_list_curr,
                                         alpha=alpha_coef_adpt)

                if algorithm == "feddynmlb":
                    alpha_dyn = setting["alpha_dyn"]
                    lambda12 = setting["lambda1_lambda2_"]

                    alpha_coef_adpt = alpha_dyn / weight_list[sampled_clients[c]]  # adaptive alpha coef
                    local_param_list_curr = load_local_param_curr_from_file(sampled_clients[c], client_list,
                                                                            server_model)

                    opt = tf.keras.optimizers.experimental.SGD(learning_rate=lr_client, momentum=momentum,
                                                               clipnorm=clipnorm,
                                                               weight_decay=(alpha_coef_adpt + l2_weight_decay))

                    local_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                                                       l2_weight_decay=0.0, seed=random_seed)

                    server_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                                                        l2_weight_decay=0.0, seed=random_seed)

                    local_model_mlb.set_weights(server_model.get_weights())
                    server_model_mlb.set_weights(server_model.get_weights())

                    client_model = FedDynMLBModel(local_model_mlb, server_model_mlb)
                    client_model.compile(optimizer=opt,
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(
                                             reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                         lambda_1=lambda12[0],
                                         lambda_2=lambda12[1],
                                         avg_mdl_param=server_model.get_weights(),
                                         local_grad_vector=local_param_list_curr,
                                         alpha=alpha_coef_adpt
                                         )

                if algorithm == "moon":
                    mu = setting["mu_moon"]
                    t = setting["t_moon"]
                    local_model_moon = create_model_moon(architecture=architecture, num_classes=num_classes, norm=norm,
                                                         l2_weight_decay=l2_weight_decay, seed=random_seed)
                    local_model_moon.set_weights(server_model.get_weights())

                    server_model_moon = create_model_moon(architecture=architecture, num_classes=num_classes, norm=norm,
                                                          l2_weight_decay=0.0, seed=random_seed)
                    server_model_moon.set_weights(server_model.get_weights())

                    last_model_moon = create_model_moon(architecture=architecture, num_classes=num_classes, norm=norm,
                                                        l2_weight_decay=0.0, seed=random_seed)

                    last_model_moon.set_weights(load_local_param_curr_from_file(client_id=sampled_clients[c],
                                                                                c_list=client_list,
                                                                                server_model=server_model,
                                                                                algorithm="moon",
                                                                                random_seed=random_seed))

                    client_model = MoonModel(local_model_moon, global_model=server_model_moon,
                                             last_model=last_model_moon)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         mu=mu,
                                         temperature=t)

                if algorithm == "fedgkd":
                    gamma = setting["gamma"]
                    client_model = FedGKDModel(resnet_model, historical_ensemble_model)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(),
                                         gamma=gamma)

                    # teacher_logits = historical_ensemble_model.predict(training_dataset)
                    # teacher_logits_2 = historical_ensemble_model.predict(training_dataset)
                    # print("logits--------")
                    # print(teacher_logits[0]-teacher_logits_2[0])
                    # teacher_logits_ds = tf.data.Dataset.from_tensor_slices(teacher_logits).batch(local_batch_size)
                    # training_dataset = tf.data.Dataset.zip((training_dataset,
                    #                                         teacher_logits_ds))

                if algorithm == "fedntd":
                    beta = setting["beta"]
                    client_model = FedNTDModel(resnet_model, server_model)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(),
                                         beta=beta
                                         )

                    # teacher_logits = server_model.predict(training_dataset)
                    # teacher_logits_ds = tf.data.Dataset.from_tensor_slices(teacher_logits).batch(local_batch_size)
                    # training_dataset = tf.data.Dataset.zip((training_dataset,
                    #                                         teacher_logits_ds))

                if algorithm == "fedmlb":
                    lambda12 = setting["lambda1_lambda2_"]
                    # tf.print("Initializing mlb model..")
                    # local_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                    #                                    l2_weight_decay=l2_weight_decay, seed=random_seed)
                    local_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                                                       l2_weight_decay=0.0, seed=random_seed)
                    # server_model_mlb = res.create_resnet8_mlb(num_classes=num_classes)
                    server_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                                                        l2_weight_decay=0.0, seed=random_seed)

                    local_model_mlb.set_weights(server_model.get_weights())
                    server_model_mlb.set_weights(server_model.get_weights())

                    client_model = FedMLB2Model(local_model_mlb, server_model_mlb)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm,
                                                                        weight_decay=l2_weight_decay),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(
                                             reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                         lambda_1=lambda12[0],
                                         lambda_2=lambda12[1]
                                         )
                    # tf.print("Done..")

                if algorithm == "fedlgic":
                    lambda12 = setting["lambda1_lambda2_"]
                    # tf.print("Initializing lgic model..")
                    local_model_lgic = create_model_lgic(architecture=architecture, num_classes=num_classes, norm=norm,
                                                         l2_weight_decay=l2_weight_decay, seed=random_seed)
                    # server_model_mlb = res.create_resnet8_mlb(num_classes=num_classes)
                    server_model_lgic = create_model_lgic(architecture=architecture, num_classes=num_classes, norm=norm,
                                                          l2_weight_decay=0.0, seed=random_seed)
                    local_model_lgic.set_weights(server_model.get_weights())
                    server_model_lgic.set_weights(server_model.get_weights())

                    client_model = FedLGICModel(local_model_lgic, server_model_lgic)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.MeanSquaredError(),
                                         lambda_1=lambda12[0],
                                         lambda_2=lambda12[1]
                                         )
                    # tf.print("Done..")

                if algorithm == "fedlgicd":
                    lambda12 = setting["lambda1_lambda2_"]
                    # tf.print("Initializing lgic model..")
                    local_model_lgic = create_model_lgic(architecture=architecture, num_classes=num_classes, norm=norm,
                                                         l2_weight_decay=l2_weight_decay, seed=random_seed)
                    # server_model_mlb = res.create_resnet8_mlb(num_classes=num_classes)
                    server_model_lgic = create_model_lgic(architecture=architecture, num_classes=num_classes, norm=norm,
                                                          l2_weight_decay=0.0, seed=random_seed)
                    local_model_lgic.set_weights(server_model.get_weights())
                    server_model_lgic.set_weights(server_model.get_weights())

                    client_model = FedLGICDModel(local_model_lgic, server_model_lgic)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         kd_loss=tf.keras.losses.KLDivergence(),
                                         lambda_1=lambda12[0],
                                         lambda_2=lambda12[1]
                                         )
                    # tf.print("Done..")

                if rnd in reporting_rounds:
                    start_time = time.time()

                if algorithm not in ["feddlb"]:
                    history = client_model.fit(
                        training_dataset,
                        batch_size=local_batch_size,
                        epochs=local_epochs,
                    )
                else:
                    pre_data = (tf.zeros([50, 32, 32, 3], tf.float32), tf.fill([50, 1], tf.cast(-1.0, tf.int64)))
                    pre_out = tf.fill([50, 1], tf.cast(-1.0, tf.int64))
                    alpha_dlb = setting["alpha_dlb"]
                    temperature = setting["temperature"]
                    for epoch in range(local_epochs):
                        print("Epoch ", epoch)
                        pre_data, pre_out = train_one_epoch(model=client_model,
                                                            train_dataset=training_dataset,
                                                            loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(
                                                                from_logits=True),
                                                            kd_loss=tf.keras.losses.KLDivergence(
                                                                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                                            pre_data=pre_data,
                                                            pre_out=pre_out,
                                                            alpha=tf.constant(alpha_dlb, tf.float32),
                                                            temperature=tf.constant(temperature, tf.float32))

                if rnd in reporting_rounds:
                    exec_time = round(time.time() - start_time, 2)
                    exec_time_mean_round = exec_time_mean_round + exec_time / k

                # print("------------------")
                # print("RUNNING VAR: ", client_model.get_running_var())
                if algorithm in ["feddyn", "feddynmlb"]:
                    # curr_model_par = tf.nest.map_structure(lambda a: a,
                    #                                        client_model.get_weights())

                    # local_param = tf.nest.map_structure(lambda a, b: a - b,
                    #                                     client_model.get_weights(), # curr_model_par
                    #                                     cld_mdl_param,
                    #                                     )

                    local_param_to_save = tf.nest.map_structure(
                        lambda client_ww, cld_ww, ww_curr: ww_curr + client_ww - cld_ww,
                        client_model.get_weights(),  # curr_model_par
                        cld_mdl_param,
                        local_param_list_curr,
                    )

                    # local_param_to_save = tf.nest.map_structure(lambda a, b: a + b,
                    #                                                              local_param_list_curr,
                    #                                                              local_param,
                    #                                                              )
                    save_local_param_to_file(sampled_clients[c], local_param_to_save, client_list)

                    # -----------------
                    # clnt_params_list[sampled_clients[c]] = curr_model_par
                    active_clnt_avg = tf.nest.map_structure(lambda a, b: a + b / k,
                                                            active_clnt_avg,
                                                            client_model.get_weights(),  # curr_model_par
                                                            )
                    if all_clients_model:
                        all_clnt_avg = tf.nest.map_structure(lambda a, b: a + b / total_clients,
                                                             all_clnt_avg,
                                                             client_model.get_weights(),  # curr_model_par
                                                             )
                    # local_param_avg = tf.nest.map_structure(lambda a, b: a + b / total_clients,
                    #                                     local_param_avg,
                    #                                     local_param_to_save,
                    #                                     )
                    # computing average efficiently
                    local_param_avg = tf.nest.map_structure(lambda average, old_value,
                                                                   new_value: average - old_value / total_clients + new_value / total_clients,
                                                            local_param_avg,
                                                            local_param_list_curr,
                                                            local_param_to_save
                                                            )

                if algorithm == "moon":
                    save_local_param_to_file(client_id=sampled_clients[c], local_param=client_model.get_weights(),
                                             c_list=client_list, algorithm="moon")

                # log on tensorboard

                if local_test:
                    client_test_ds = data_utility.load_client_datasets_from_files(
                        dataset=dataset,
                        sampled_client=sampled_clients[c],
                        batch_size=local_batch_size,
                        alpha=alpha,
                        split="test")

                    loss, accuracy = client_model.evaluate(client_test_ds)

                    mean_client_loss_lda = mean_client_loss_lda + loss / k
                    mean_client_accuracy_lda = mean_client_accuracy_lda + accuracy / k

                    loss, accuracy = client_model.evaluate(test_ds)

                    mean_client_loss = mean_client_loss + loss / k
                    mean_client_accuracy = mean_client_accuracy + accuracy / k

                    # logging.write_logs("client_local_training_history", history.history, algorithm, client=c)

                if algorithm not in ["feddistill", "feddyn", "moon", "feddynmlb"]:
                    if norm in ["batch"]:
                        delta_w_local_with_batch = tf.nest.map_structure(lambda a, b: a - b,
                                                                         client_model.get_weights(),
                                                                         server_model.get_weights(),
                                                                         )

                        delta_w_with_batch = tf.nest.map_structure(
                            lambda a, b: a + b * (local_examples / total_examples),
                            delta_w_with_batch,
                            delta_w_local_with_batch)
                    else:
                        # print(" ---here")
                        # time.sleep(2)
                        # problem here!
                        # client_model.summary()
                        # server_model.summary()
                        # time.sleep(100)
                        delta_w_local_trainable = tf.nest.map_structure(lambda a, b: a - b,
                                                                        client_model.trainable_variables,
                                                                        server_model.trainable_variables,
                                                                        )
                        # print(" ---here")
                        # time.sleep(2)
                        delta_w_global_trainable = tf.nest.map_structure(
                            lambda a, b: a + b * (local_examples / total_examples),
                            delta_w_global_trainable,
                            delta_w_local_trainable)
                # print(" ---here")
                # time.sleep(2)
                if algorithm in ["fedfa"]:
                    # in fedfa there are non trainable params
                    fedfa_non_train_global_mean.append(client_model.non_trainable_weights[0])
                    fedfa_non_train_global_std.append(client_model.non_trainable_weights[1])
                    # fedfa_non_train_global_mean.append(client_model.get_running_var()[0])
                    # fedfa_non_train_global_std.append(client_model.get_running_var()[1])


                reset_tensorflow_keras_backend()
                del client_model
                # del resnet_model
                # del opt
            # server_model.set_weights(new_w_global)
            if algorithm not in ["feddyn", "feddynmlb"]:
                if norm in "batch":
                    new_ww = tf.nest.map_structure(lambda a, b: a + b,
                                                   server_model.get_weights(),
                                                   delta_w_with_batch)
                    server_model.set_weights(new_ww)
                else:
                    gradients = tf.nest.map_structure(lambda a: -a, delta_w_global_trainable)
                    server_model.optimizer.apply_gradients(zip(gradients, server_model.trainable_variables))

                if algorithm in ["fedfa"]:
                    # Aggregating the non trainable params of fedfa's additional layers
                    aggregated_non_train_global_mean = tf.nest.map_structure(lambda a:
                                                                             fedfa_utility.aggregate_clients_var(a),
                                                                             fedfa_non_train_global_mean)

                    aggregated_non_train_global_std = tf.nest.map_structure(lambda a:
                                                                            fedfa_utility.aggregate_clients_var(a),
                                                                            fedfa_non_train_global_std)
                    # print("aggregated_non_train_global_mean[0] ", aggregated_non_train_global_mean[0])
                    # print("aggregated_non_train_global_std[0] ", aggregated_non_train_global_std[0])

                # if setting["SWA"] == True:
                #     cycle_lenght = 20  # tested in [5, 10, 20]
                #     if rnd == 0.75 * total_rounds:
                #         swa_w = tf.nest.map_structure(lambda a: a,
                #                                       server_model.get_weights())
                #
                #     if rnd >= 0.75 * total_rounds and np.mod(rnd, cycle_lenght) == 0:
                #         n_models = rnd / cycle_lenght
                #         swa_w = tf.nest.map_structure(lambda a, b: ((a * n_models) + b) / (n_models + 1),
                #                                       swa_w,
                #                                       server_model.get_weights())
                #         server_model.set_weights(swa_w)
                # Missing local lr scheduling

            if algorithm in ["feddyn", "feddynmlb"]:
                # average of client model params init
                # avg_mdl_param = tf.nest.map_structure(lambda a, b: a - b,
                #                                       server_model.get_weights(),
                #                                       server_model.get_weights())
                # print("selected clients ", [i for i in sampled_clients])
                # temp = [clnt_params_list[i] for i in sampled_clients]
                # # print("temp len ", len(temp))
                # # print(tf.reduce_sum(temp[0][0]))
                #
                # for ww in temp:
                #     avg_mdl_param = tf.nest.map_structure(lambda a, b: a + b / len(temp),
                #                                           avg_mdl_param,
                #                                           ww)

                # here modified
                # local_param_avg = tf.nest.map_structure(lambda a, b: a - b,
                #                                         server_model.get_weights(),
                #                                         server_model.get_weights())
                #
                # for ww in local_param_list:
                #     local_param_avg = tf.nest.map_structure(lambda a, b: a + b / total_clients,
                #                                             local_param_avg,
                #                                             ww)
                # -------------

                # add this mean
                cld_mdl_param = tf.nest.map_structure(lambda a, b: a + b,
                                                      active_clnt_avg,
                                                      local_param_avg)

                server_model.set_weights(cld_mdl_param)
                # cld_mdl_param = avg_mdl_param + local_param_avg

                # avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis=0)
                # cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
                # all_mdl_param = tf.nest.map_structure(lambda a, b: a - b,
                #                                       server_model.get_weights(),
                #                                       server_model.get_weights())
                # for ww in clnt_params_list:
                #     all_mdl_param = tf.nest.map_structure(lambda a, b: a + b / len(clnt_params_list),
                #                                           all_mdl_param,
                #                                           ww)

                if rnd == interrupting_round:
                    path = os.path.join(algorithm + "_checkpoints", "server_model", "saved_local_param")
                    server_model.save_weights(path)
                    path = os.path.join(algorithm + "_checkpoints", "local_param_avg_model", "local_param_avg")
                    server_model.set_weights(local_param_avg)
                    server_model.save_weights(path)

            if algorithm in ["fedsam", "fedfa", "feddlb"]:
                if rnd == interrupting_round:
                    path = os.path.join(algorithm + "_checkpoints", "server_model", "saved_local_param")
                    server_model.save_weights(path)
                    opt_dict = server_model.optimizer.get_config()
                    path = os.path.join(algorithm + "_checkpoints", "opt_dict")


                    def dic_save(dictionary, filename):
                        # with open(filename + '.json', 'w') as fp:
                        #     json.dump(dictionary, fp)
                        with open(filename + '.pickle', 'wb') as f:
                            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


                    dic_save(opt_dict, path)
                    if algorithm == "fedfa":
                        fedfa_dict = {"aggregated_mean": aggregated_non_train_global_mean,
                                      "aggregated_std": aggregated_non_train_global_mean}
                        path = os.path.join(algorithm + "_checkpoints", "mean_std")
                        dic_save(fedfa_dict, path)

            if algorithm == "fedgkd":
                # print("fedgkd len ", len(historical_weights))
                if len(historical_weights) == M:
                    historical_weights.pop(0)
                historical_weights.append(server_model.get_weights())

                # Compute ensemble of M historical global models
                # Initialize the ensemble to 0
                historical_ensemble_weights = tf.nest.map_structure(lambda a, b: a - b,
                                                                    historical_weights[0],
                                                                    historical_weights[0])
                for model_weights in historical_weights:
                    historical_ensemble_weights = tf.nest.map_structure(lambda a, b: a + b / len(historical_weights),
                                                                        historical_ensemble_weights,
                                                                        model_weights)

                historical_ensemble_model.set_weights(historical_ensemble_weights)
                # history = historical_ensemble_model.evaluate(test_ds, return_dict=True)
                # with historical_summary_writer.as_default():
                #     tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                #     tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)

            # print(f"[info] mean client loss {mean_client_loss}")
            # print(f"[info] mean client accuracy {mean_client_accuracy}")
            # logging.write_logs("average_client_evaluation_after_local_training",
            #                    "accuracy " + str(mean_client_accuracy) + " loss " + str(mean_client_loss), algorithm)

            if local_test:
                with local_test_summary_writer.as_default():
                    tf.summary.scalar('loss', mean_client_loss, step=rnd)
                    tf.summary.scalar('accuracy', mean_client_accuracy, step=rnd)

                with local_test_lda_summary_writer.as_default():
                    tf.summary.scalar('loss', mean_client_loss_lda, step=rnd)
                    tf.summary.scalar('accuracy', mean_client_accuracy_lda, step=rnd)

            print("[Server] Evaluation - Round: ", rnd)
            if algorithm in ["feddyn", "feddynmlb"]:
                server_model.set_weights(active_clnt_avg)
            history = server_model.evaluate(test_ds, return_dict=True)
            with global_summary_writer.as_default():
                tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)

            if algorithm in ["feddyn", "feddynmlb"]:
                if all_clients_model:
                    server_model.set_weights(all_clnt_avg)
                    history = server_model.evaluate(test_ds, return_dict=True)
                    with global_dyn_summary_writer.as_default():
                        tf.summary.scalar('loss_all', tf.squeeze(history["loss"]), step=rnd)
                        tf.summary.scalar('accuracy_all', tf.squeeze(history["accuracy"]), step=rnd)

                server_model.set_weights(cld_mdl_param)
                # history = server_model.evaluate(test_ds, return_dict=True)
                # with global_summary_writer.as_default():
                #     tf.summary.scalar('loss_cld', tf.squeeze(history["loss"]), step=rnd)
                #     tf.summary.scalar('accuracy_cld', tf.squeeze(history["accuracy"]), step=rnd)

            if rnd in reporting_rounds:
                exec_time_mean_reporting_rounds = exec_time_mean_reporting_rounds + exec_time_mean_round / len(
                    reporting_rounds)

            # logging.write_logs("server_evaluation", history, algorithm)
            # lr_client = exp_decayed_learning_rate(initial_learning_rate=lr_client_initial, decay_rate=exp_decay,
            #                                       decay_steps=total_rounds, step=rnd)
            lr_client *= exp_decay

        if reporting_exec_time_enabled:
            logging.write_intro_logs(folder="monitoring_exec_time", list_file_names=[algorithm + "_exec_time"],
                                     dict_sett=setting)
            logging.write_logs(folder="monitoring_exec_time", file_name=algorithm + "_exec_time",
                               history={algorithm: exec_time_mean_reporting_rounds,
                                        "reporting_rounds": str(reporting_rounds)})
