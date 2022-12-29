import os

# This is for explicitly use GPU recognized as device 0
# Remove this line if you dont need it
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import data_utility
import tensorflow as tf
import numpy as np
from tensorflow import keras
# import residual_model as res
# import residual_model as res
import residual_model_custom as res
from non_iid_algorithms.FedGKD2Model import FedGKDModel
from non_iid_algorithms.FedNTDModel import FedNTDModel
from non_iid_algorithms.FedProxModel import FedProxModel
from non_iid_algorithms.FedLGICModel import FedLGICModel
from non_iid_algorithms.FedMLB2Model import FedMLB2Model
import tensorflow_datasets as tfds
import logging_utility as logging
from typing import Optional
from distiller import Distiller
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        device=gpu, enable=True
    )

PATH = os.path.join("results_1811_distill")


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


def create_model(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    # return res.create_resnet18(input_shape=(32, 32, 3), num_classes=10, norm='batch')
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
        return keras.optimizers.Adam(learning_rate=lr)

# def exp_decayed_learning_rate(initial_learning_rate, decay_rate, step, decay_steps):
#     return initial_learning_rate * decay_rate ** (step / decay_steps)

def distill(model, image_pred, epochs=5, client=None):
    # distillation
    distiller = Distiller(student=model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        # temperature=0.1,  # era aggregation
        metrics=[tf.keras.metrics.KLDivergence()]
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_kullback_leibler_divergence",
        patience=1,
        verbose=0,
        restore_best_weights=True,
    )

    # Distill teacher to student
    h = distiller.fit(image_pred, epochs=epochs, callbacks=[callback], validation_data=image_pred)

    return distiller

if __name__ == '__main__':
    # Building a dictionary of hyperparameters
    hp = {}
    hp["algorithm"] = ["fedavg"]
    hp["reinit_classifier"] = [False]
    hp["batch_size"] = [64]
    hp["E"] = [10]  # local_epochs
    hp["C"] = [8]  # n clients
    hp["total_clients"] = [20]
    hp["rounds"] = [12]
    hp["alpha"] = [100.0, 1.0]
    hp["norm"] = ["group"]
    hp["dataset"] = ["cifar10"]
    hp["lr_client"] = [0.1]
    hp["momentum"] = [0.0]
    hp["weight_decay"] = [1e-4]
    hp["architecture"] = ["resnet8"]
    hp["server_side_optimizer"] = ["adam"]
    hp["lr_server"] = [0.001, 0.0001]
    # hp["lr_server"] = [1.0, 1.5, 0.1, 0.01]
    hp["server_momentum"] = [0.0]
    hp["lr_decay"] = [0.998]
    hp["v"] = [1]

    hp["seed"] = [1994]  # seed for client selection

    hp["aggregation"] = ["mean_probabilities"]
    # hp["batch_size"] = [32, 16, 64]
    hp["distill_batch_size"] = [128]
    hp["max_distill_epochs_server"] = [20]
    hp["max_distill_epochs_clients"] = [20]
    hp["distill_examples"] = [60000]
    # possible values for transfer set
    # cifar10, stl10, svhn, sop, random, tiny
    # tiny dataset is not here yet
    hp["transfer_set"] = ["cifar10"]

    # algorithm-specific hyperparameter
    cd = {}
    cd["codist"] = {"v": ["1"]}
    # cd["fedgkd"] = {"M": [1, 5], "gamma": [0.2]}
    # cd["fedlgic"] = {"lambda1_lambda2_": [[0.7, 0.7]]}
    # cd["fedprox"] = {"mu": [0.001, 0.01]}
    # cd["fedntd"] = {"beta": [0.1, 0.3, 0.7]}
    # cd["fedmlb"] = {"lambda1_lambda2_": [[0.01, 0.05]]}
    # cd["fedlgic"] = {"lambda1_lambda2_": [[0.7, 0.7]]}
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

        batch_size_distill = setting["distill_batch_size"]
        distillation_size = setting["distill_examples"]
        distillation_epochs = setting["max_distill_epochs_server"]
        clients_distillation_epochs = setting["max_distill_epochs_clients"]
        aggregation = setting["aggregation"]
        reinit_classifier = setting["reinit_classifier"]
        transfer_set = setting["transfer_set"]
        co_distillation = True
        num_classes = 10
        if dataset == "cifar100":
            num_classes = 100

        # Some log files for metrics and debug info
        # logging.write_intro_logs(["server_evaluation", "client_local_training_history",
        #                           "average_client_evaluation_after_local_training"], setting, algorithm)

        lr_client = lr_client_initial
        server_model = create_model(architecture=architecture, num_classes=num_classes, norm=0.0, seed=random_seed)

        server_optimizer = create_server_optimizer(optimizer=server_side_optimizer, lr=lr_server,
                                                   momentum=server_momentum)

        server_model.compile(
            optimizer=server_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )
        server_model.summary()

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
        # ..../dataset_alpha--_C--_k--/algorithm/general_hyperparameters/specific_hyperparameters,seed---
        logdir = os.path.join(PATH, dataset + "_alpha" + str(round(alpha, 2)) + "_C" + str(round(total_clients, 2)) +
                              "_k" + str(round(k, 2)), architecture,
                              logging.encode_general_hyperparameter_in_string(setting),
                              algorithm,
                              logging.encode_specific_hyperparameter_in_string(setting) + "," + "seed" + str(
                                  round(random_seed, 2)))

        global_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "global_test"))

        history = server_model.evaluate(test_ds, return_dict=True)
        with global_summary_writer.as_default():
            tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=0)
            tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=0)

        seed_x: int = random.randint(0, 1000)
        for rnd in range(1, total_rounds + 1):
            # cifar10 is partitioned among 20 clients
            list_of_clients = [i for i in range(0, total_clients)]
            sampled_clients = np.random.choice(
                list_of_clients,
                size=k,
                replace=False)
            print("Selected clients for the next round ", sampled_clients)

            global_weights = server_model.get_weights()

            delta_w_init = tf.nest.map_structure(lambda a, b: a - b,
                                                 global_weights,
                                                 global_weights)

            selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)
            global_predictions = tf.zeros([distillation_size, num_classes], tf.float32)

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
                    alpha=alpha)

                client_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                            l2_weight_decay=l2_weight_decay, seed=random_seed)

                if rnd > 1:
                    client_model.set_weights(client_weights)

                client_model.compile(optimizer=keras.optimizers.Adam(),
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                     )
                # client_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_client, momentum=momentum),
                #                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                #                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                #                      )

                history = client_model.fit(
                    training_dataset,
                    batch_size=local_batch_size,
                    epochs=local_epochs,
                )

                # local predictions
                transfer_set_ds = data_utility.load_transfer_set(
                    transfer_set,
                    num_examples=distillation_size,
                    seed=seed_x)

                logits = client_model.predict(transfer_set_ds.batch(batch_size=1024))

                if aggregation == "mean_probabilities":
                    probabilities = tf.nn.softmax(logits, axis=1)
                    global_predictions = global_predictions + (probabilities / total_examples) * local_examples

            aggregated_predictions = global_predictions

            # Co-distillation
            if co_distillation:
                # seed_y = random.randint(0, 19)
                print("[Server] Distillation ---")

                transfer_set_ds = data_utility.load_transfer_set(
                    transfer_set,
                    num_examples=distillation_size,
                    seed=seed_x)

                teacher_predictions = tf.data.Dataset.from_tensor_slices(aggregated_predictions)
                img_pred = zip_img_and_pred(transfer_set_ds, teacher_predictions, batch_size_distill=batch_size_distill)

                server_weights_before = tf.nest.map_structure(lambda a: a,
                                                        server_model.get_weights()
                                                      )

                server_distiller = distill(server_model, img_pred, distillation_epochs)
                server_weights = server_distiller.get_weights()
                # server_model already have that weights?
                # optimizer
                delta_w_global = tf.nest.map_structure(lambda a, b: a - b,
                                                          server_weights,
                                                          server_weights_before,
                                                          )

                gradients = tf.nest.map_structure(lambda a: -a, delta_w_global)
                server_model.set_weights(server_weights_before)
                server_model.optimizer.apply_gradients(zip(gradients, server_model.trainable_variables))
                # server_model.set_weights(server_weights)

                transfer_set_ds = data_utility.load_transfer_set(
                    transfer_set,
                    num_examples=distillation_size,
                    seed=seed_x)

                print("[Server] Predictions ---")
                server_logits = server_model.predict(transfer_set_ds.batch(batch_size=1024))
                aggregated_predictions = tf.nn.softmax(server_logits, axis=1)

                print("[Server] Evaluation - Round: ", rnd)
                history = server_model.evaluate(test_ds, return_dict=True)

                with global_summary_writer.as_default():
                    tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                    tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)

                # lr_client *= exp_decay

            # Client distillation
            # Since in this simulation we are using the same architecture for all clients
            # We can distill once for all clients
            # and having them start from the same model weights
            # This reduces the time for simulations!
            print(f"[Client] Distillation ---")
            client_model = create_model(architecture=architecture, num_classes=num_classes, norm=norm,
                                            l2_weight_decay=l2_weight_decay, seed=random_seed)

            # client_model.compile(optimizer=keras.optimizers.Adam(),
            #                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

            teacher_predictions = tf.data.Dataset.from_tensor_slices(aggregated_predictions)
            transfer_set_ds = data_utility.load_transfer_set(
                transfer_set,
                num_examples=distillation_size,
                seed=seed_x)

            img_pred = zip_img_and_pred(transfer_set_ds, teacher_predictions, batch_size_distill=batch_size_distill)
            client_distiller = distill(client_model, img_pred, clients_distillation_epochs)
            client_weights = client_distiller.get_weights()
            seed_x: int = random.randint(0, 1000)
