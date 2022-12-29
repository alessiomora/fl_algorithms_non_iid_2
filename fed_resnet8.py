import os

# This is for explicitly use GPU recognized as device 0
# Remove this line if you dont need it
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import data_utility
import tensorflow as tf
import numpy as np
from tensorflow import keras
# import residual_model as res
import residual_model_custom as res
from non_iid_algorithms.FedDynModel import FedDynModel
from non_iid_algorithms.FedGKDModel import FedGKDModel
from non_iid_algorithms.FedNTDModel import FedNTDModel
from non_iid_algorithms.FedProxModel import FedProxModel
from non_iid_algorithms.FedLGICModel import FedLGICModel
from non_iid_algorithms.FedLGICDModel import FedLGICDModel
from non_iid_algorithms.FedMLB2Model import FedMLB2Model
import tensorflow_datasets as tfds
import logging_utility as logging
from typing import Optional

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        device=gpu, enable=True
    )

PATH = os.path.join("determinism_2")


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


# def exp_decayed_learning_rate(initial_learning_rate, decay_rate, step, decay_steps):
#     return initial_learning_rate * decay_rate ** (step / decay_steps)


if __name__ == '__main__':
    # Building a dictionary of hyperparameters
    hp = {}
    # hp["algorithm"] = ["fedavg", "fedgkd", "fedprox", "fedntd", "fedmlb"]
    hp["algorithm"] = ["feddyn"]
    hp["reinit_classifier"] = [False]
    hp["batch_size"] = [50]
    hp["E"] = [5]  # local_epochs
    hp["C"] = [5]  # n clients
    hp["total_clients"] = [100]
    hp["rounds"] = [1000]
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
    hp["v2_lastdecay_2"] = [50]

    hp["seed"] = [0]  # seed for client selection, model initialization, client data shuffling

    # what can I do then:

    # ------------------------------
    # algorithm-specific hyperparameter
    cd = {}
    # cd["fedgkd"] = {"M": [1, 5], "gamma": [0.2]}
    # cd["fedlgic"] = {"lambda1_lambda2_": [[0.7, 0.7], [1.0, 1.0]]}
    # cd["fedlgicd"] = {"lambda1_lambda2_": [[0.1, 0.1], [0.15, 0.15]]}
    # cd["fedprox"] = {"mu": [0.01, 0.001]}
    cd["feddyn"] = {"alpha_dyn": [0.1]}
    # cd["fedntd"] = {"beta": [1.0, 0.3]}
    # cd["fedmlb"] = {"lambda1_lambda2_": [[1.0, 1.0]]}
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
        local_batch_size = setting["batch_size"]
        test_batch_size = 256
        reinit_classifier = setting["reinit_classifier"]
        alpha = setting["alpha"]
        algorithm = setting["algorithm"]
        norm = setting["norm"]
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

        local_test = False

        if dataset == "cifar100":
            num_classes = 100

        # Some log files for metrics and debug info
        # logging.write_intro_logs(["server_evaluation", "client_local_training_history",
        #                           "average_client_evaluation_after_local_training"], setting, algorithm)

        lr_client = lr_client_initial
        server_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm, seed=random_seed)

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

        if algorithm == "feddyn":
            weight_list = np.asarray([500.0 for i in range(100)])
            weight_list = weight_list / np.sum(weight_list) * total_clients
            print("weight list", weight_list)
            zeros_ww = tf.nest.map_structure(lambda a, b: a - b,
                                             server_model.get_weights(),
                                             server_model.get_weights())

            local_param_list = [zeros_ww for i in range(total_clients)]

            init_ww = tf.nest.map_structure(lambda a: a,
                                            server_model.get_weights())

            clnt_params_list = [init_ww for i in range(total_clients)]

            cld_mdl_param = tf.nest.map_structure(lambda a: a,
                                                  server_model.get_weights(),
                                                  )

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
        global_dyn_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "all_test"))

        # historical_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "historical_test"))

        history = server_model.evaluate(test_ds, return_dict=True)
        with global_summary_writer.as_default():
            tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=0)
            tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=0)

        for rnd in range(1, total_rounds + 1):
            # cifar10 is partitioned among 20 clients
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

            # if algorithm != "feddyn" or (algorithm == "feddyn" and rnd == 1):
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
            #
            # w_init_non_trainable = tf.nest.map_structure(lambda a, b: a - b,
            #                                      server_model.non_trainable_weights,
            #                                      server_model.non_trainable_weights)

            if algorithm == "feddistill_plus" or algorithm == "feddistill":
                global_mean_per_label_soft_labels = tf.zeros([10, 10], tf.float32)

            selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)

            for c in range(0, k):
                print("Client: ", c)
                local_examples = selected_client_examples[c]
                print("Local examples: ", local_examples)

                # Local training
                print(f"[Client {c}] Local Training ---")

                training_dataset = data_utility.load_client_datasets_from_files(
                    dataset=dataset,
                    sampled_client=sampled_clients[c],
                    batch_size=local_batch_size,
                    alpha=alpha,
                    seed=random_seed)

                if algorithm != "feddyn":
                    resnet_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                l2_weight_decay=l2_weight_decay, seed=random_seed)

                if algorithm != "feddistill" and algorithm != "feddyn":  # fed_distill does not exchange weights
                    resnet_model.set_weights(aggregated_weights)

                if algorithm == "fedavg":
                    client_model = resnet_model
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         )
                    # client_model.summary()

                if algorithm == "fedprox":
                    mu = setting["mu"]
                    client_model = FedProxModel(resnet_model)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         initial_weights=aggregated_weights,
                                         mu=mu)

                if algorithm == "feddyn":
                    alpha_dyn = setting["alpha_dyn"]
                    alpha_coef_adpt = alpha_dyn / weight_list[sampled_clients[c]]  # adaptive alpha coef

                    resnet_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                                # l2_weight_decay=(alpha_coef_adpt + l2_weight_decay),
                                                l2_weight_decay=0.0,
                                                seed=random_seed)
                    resnet_model.set_weights(aggregated_weights)

                    local_param_list_curr = local_param_list[sampled_clients[c]]
                    client_model = FedDynModel(resnet_model)
                    opt = tf.keras.optimizers.experimental.SGD(learning_rate=lr_client, momentum=momentum,
                                                               clipnorm=clipnorm,
                                                               weight_decay=(alpha_coef_adpt + l2_weight_decay))
                    # opt = keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum, clipnorm=clipnorm)
                    client_model.compile(optimizer=opt,
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                         avg_mdl_param=aggregated_weights,
                                         local_grad_vector=local_param_list_curr,
                                         alpha=alpha_coef_adpt)

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
                    # local_model_mlb = res.create_resnet8_mlb(num_classes=num_classes, l2_weight_decay=l2_weight_decay)
                    local_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                                                       l2_weight_decay=l2_weight_decay, seed=random_seed)
                    # server_model_mlb = res.create_resnet8_mlb(num_classes=num_classes)
                    server_model_mlb = create_model_mlb(architecture=architecture, num_classes=num_classes, norm=norm,
                                                        l2_weight_decay=0.0, seed=random_seed)

                    local_model_mlb.set_weights(aggregated_weights)
                    server_model_mlb.set_weights(aggregated_weights)

                    client_model = FedMLB2Model(local_model_mlb, server_model_mlb)
                    client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum,
                                                                        clipnorm=clipnorm),
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

                history = client_model.fit(
                    training_dataset,
                    batch_size=local_batch_size,
                    epochs=local_epochs,
                )

                if algorithm == "feddyn":
                    curr_model_par = tf.nest.map_structure(lambda a: a,
                                                           client_model.get_weights())
                    local_param = tf.nest.map_structure(lambda a, b: a - b,
                                                        curr_model_par,
                                                        cld_mdl_param,
                                                        )
                    print("local_param ", tf.reduce_sum(local_param[0]))
                    # This could not work
                    # local_param_list[sampled_clients[c]] += local_param
                    local_param_list[sampled_clients[c]] = tf.nest.map_structure(lambda a, b: a + b,
                                                                                 local_param_list[sampled_clients[c]],
                                                                                 local_param,
                                                                                 )

                    clnt_params_list[sampled_clients[c]] = curr_model_par

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

                if algorithm != "feddistill" and algorithm != "feddyn":
                    # global_weights = tf.nest.map_structure(lambda a, b: a + (local_examples * b) / total_examples,
                    #                                        global_weights,
                    #                                        client_model.model.get_weights() if algorithm not in [
                    #                                            "fedavg", "fedprox", "fedmlb", "fedlgic"] else
                    #                                        client_model.get_weights())

                    # ---- here -----
                    delta_w_local_trainable = tf.nest.map_structure(lambda a, b: a - b,
                                                                    client_model.get_weights(),
                                                                    server_model.get_weights(),
                                                                    )
                    delta_w_global_trainable = tf.nest.map_structure(
                        lambda a, b: a + b * (local_examples / total_examples),
                        delta_w_global_trainable,
                        delta_w_local_trainable)
                    # -------------------
                    #
                    # # Non trainable params
                    # new_w_global_non_trainable = tf.nest.map_structure(lambda a, b: a + (local_examples * b) / total_examples,
                    #                                        new_w_global_non_trainable,
                    #                                        client_model.non_trainable_weights)

                    # ---Alternative
                    # new_w_global = tf.nest.map_structure(lambda a, b: a + b * (local_examples / total_examples),
                    #                                        new_w_global,
                    #                                        client_model.get_weights())

            # server_model.set_weights(new_w_global)
            if algorithm != "feddyn":
                gradients = tf.nest.map_structure(lambda a: -a, delta_w_global_trainable)
                server_model.optimizer.apply_gradients(zip(gradients, server_model.trainable_variables))
            # for i in range(0, len(server_model.non_trainable_weights)):
            #     server_model.non_trainable_weights[i] = new_w_global_non_trainable[i]

            # print("---")
            # for layer in server_model.layers:
            #     try:
            #         for ll in layer.layers:
            #             print(ll.name)
            #             # layer_weights = []
            #             # if len(ll.trainable_weights) > 0:
            #             #     print(ll)
            #             #     layer_weights += ll.trainable_weights
            #             # if len(ll.non_trainable_weights) > 0:
            #             #     print(ll)
            #             #     layer_weights += new_w_global_non_trainable
            #             # ll.set_weights(layer_weights)
            #
            #     except:
            #         print("dense")

            # Server evaluation
            # server_model.set_weights(global_weights)

            if algorithm == "feddyn":
                # average of client model params init
                avg_mdl_param = tf.nest.map_structure(lambda a, b: a - b,
                                                      server_model.get_weights(),
                                                      server_model.get_weights())
                print("selected clients ", [i for i in sampled_clients])
                temp = [clnt_params_list[i] for i in sampled_clients]
                # print("temp len ", len(temp))
                # print(tf.reduce_sum(temp[0][0]))

                for ww in temp:
                    avg_mdl_param = tf.nest.map_structure(lambda a, b: a + b / len(temp),
                                                          avg_mdl_param,
                                                          ww)

                local_param_avg = tf.nest.map_structure(lambda a, b: a - b,
                                                        server_model.get_weights(),
                                                        server_model.get_weights())

                print("local_param_list len ", len(local_param_list))
                for ww in local_param_list:
                    local_param_avg = tf.nest.map_structure(lambda a, b: a + b / len(local_param_list),
                                                            local_param_avg,
                                                            ww)

                # add this mean
                cld_mdl_param = tf.nest.map_structure(lambda a, b: a + b,
                                                      avg_mdl_param,
                                                      local_param_avg)

                server_model.set_weights(cld_mdl_param)
                # cld_mdl_param = avg_mdl_param + local_param_avg

                # avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis=0)
                # cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
                all_mdl_param = tf.nest.map_structure(lambda a, b: a - b,
                                                      server_model.get_weights(),
                                                      server_model.get_weights())
                for ww in clnt_params_list:
                    all_mdl_param = tf.nest.map_structure(lambda a, b: a + b / len(clnt_params_list),
                                                          all_mdl_param,
                                                          ww)

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
            server_model.set_weights(avg_mdl_param)
            history = server_model.evaluate(test_ds, return_dict=True)
            with global_summary_writer.as_default():
                tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)

            if algorithm == "feddyn":
                server_model.set_weights(all_mdl_param)
                history = server_model.evaluate(test_ds, return_dict=True)
                with global_dyn_summary_writer.as_default():
                    tf.summary.scalar('loss_all', tf.squeeze(history["loss"]), step=rnd)
                    tf.summary.scalar('accuracy_all', tf.squeeze(history["accuracy"]), step=rnd)

                server_model.set_weights(cld_mdl_param)
                history = server_model.evaluate(test_ds, return_dict=True)
                with global_dyn_summary_writer.as_default():
                    tf.summary.scalar('loss_cld', tf.squeeze(history["loss"]), step=rnd)
                    tf.summary.scalar('accuracy_cld', tf.squeeze(history["accuracy"]), step=rnd)


            # logging.write_logs("server_evaluation", history, algorithm)
            # lr_client = exp_decayed_learning_rate(initial_learning_rate=lr_client_initial, decay_rate=exp_decay,
            #                                       decay_steps=total_rounds, step=rnd)
            lr_client *= exp_decay
