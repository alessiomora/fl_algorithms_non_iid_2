import os
import data_utility
import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
import residual_model as res
from distiller import Distiller
import itertools as it
import tensorflow_datasets as tfds
import logging_utility as logging

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        device=gpu, enable=True
    )
PATH = "exp_results_cifar100"


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

    logging.write_logs("clients_distill_history", h.history, "feddf", client)
    return distiller


if __name__ == '__main__':

    # Building a dictionary of hyperparameters
    hp = {}
    hp["batch_size"] = [32]
    hp["E"] = [20]  # local_epochs
    hp["C"] = [4]  # n clients
    hp["rounds"] = [100]
    hp["distill_batch_size"] = [128]
    hp["max_distill_epochs_server"] = [20]
    hp["distill_examples"] = [60000]
    hp["alpha"] = [0.1]
    hp["dataset"] = ["cifar100"]
    hp["transfer_set"] = ["tiny"]
    hp["seed"] = [2021, 2022]
    hp["norm"] = ["group"]

    # Creating a list of dictionaries
    # each one for a combination of hp
    settings = logging.get_all_hp_combinations(hp)

    # Running a simulation for each of the setting in hp possible combinations
    for setting in settings:
        print("simulation start " + str(setting))
        total_rounds = setting["rounds"]
        total_clients = setting["C"]
        local_epochs = setting["E"]
        local_batch_size = setting["batch_size"]
        test_batch_size = 256
        batch_size_distill = setting["distill_batch_size"]
        distillation_size = setting["distill_examples"]
        distillation_epochs = setting["max_distill_epochs_server"]
        random_seed = setting["seed"]
        dataset = setting["dataset"]
        transfer_set = setting["transfer_set"]
        alpha = setting["alpha"]
        norm = setting["norm"]

        lr_client = 10 ** (-0.5)

        num_classes = 10
        if dataset == "cifar100":
            num_classes = 100
        # Some log files for metrics and debug info
        logging.write_intro_logs(["server_evaluation", "client_local_training_history",
                                  "average_client_evaluation_after_local_training"], setting, "feddf")

        server_model = res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm)

        server_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )

        if dataset == "cifar10":
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(element_norm_fn).batch(
                test_batch_size)

        if dataset == "cifar100":
            cifar100_tfds = tfds.load("cifar100")
            test_ds = cifar100_tfds["test"]
            test_ds = test_ds.map(element_norm_fn_cifar100).batch(test_batch_size)

        logdir = os.path.join(PATH, "logs", dataset + "_" + str(round(alpha, 2)), "feddf",
                              logging.encode_hyperparameter_in_string(setting))
        global_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "global_test"))

        history = server_model.evaluate(test_ds, return_dict=True)
        logging.write_logs("server_evaluation", history, "feddf")
        with global_summary_writer.as_default():
            tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=0)
            tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=0)

        np.random.seed(random_seed)
        for rnd in range(1, total_rounds + 1):
            # global_predictions = tf.zeros([distillation_size, 10], tf.float32)

            # cifar10 is partitioned among 20 clients
            list_of_clients = [i for i in range(0, total_clients)]
            sampled_clients = np.random.choice(
                list_of_clients,
                size=total_clients,
                replace=False)
            print("Selected clients for the next round ", sampled_clients)
            selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)

            transfer_set_ds = data_utility.load_transfer_set(
                transfer_set,
                num_examples=distillation_size,
                seed=1)

            mean_client_loss = 0
            mean_client_accuracy = 0

            global_weights = server_model.get_weights()
            # print(global_weights[0][0])
            aggregated_weights = tf.nest.map_structure(lambda a: a,
                                                       global_weights)

            global_weights = tf.nest.map_structure(lambda a, b: a - b,
                                                   global_weights,
                                                   global_weights)
            client_predictions = []
            for c in range(0, total_clients):
                local_examples = selected_client_examples[c]
                print("Client: ", c)
                client_model = res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm)
                client_model.compile(
                    optimizer=keras.optimizers.Adam() if dataset == "cifar10" else keras.optimizers.SGD(
                        learning_rate=lr_client),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

                client_model.set_weights(aggregated_weights)

                # Local training
                print(f"[Client {c}] Local Training ---")
                # training_dataset = data_utility.load_client_cifar10_datasets_from_files(
                #     sampled_client=sampled_clients[c],
                #     batch_size=local_batch_size)

                training_dataset = data_utility.load_client_datasets_from_files(
                    dataset=dataset,
                    sampled_client=sampled_clients[c],
                    batch_size=local_batch_size,
                    alpha=alpha)

                history = client_model.fit(
                    training_dataset,
                    batch_size=local_batch_size,
                    epochs=local_epochs
                )
                logging.write_logs("client_local_training_history", history.history, "feddf", client=c)

                # global_weights = tf.nest.map_structure(lambda a, b: a + b/8,
                #                       global_weights,
                #                       client_model.get_weights())

                global_weights = tf.nest.map_structure(lambda a, b: a + (local_examples * b) / total_examples,
                                                       global_weights,
                                                       client_model.get_weights())

                # print(global_weights[0])
                loss, accuracy = client_model.evaluate(test_ds)

                mean_client_loss = mean_client_loss + loss / total_clients
                mean_client_accuracy = mean_client_accuracy + accuracy / total_clients
                preds = client_model.predict(transfer_set_ds.batch(1024))
                client_predictions.append(preds)

            # Server evaluation
            server_model.set_weights(global_weights)

            print(f"[info] mean client loss {mean_client_loss}")
            print(f"[info] mean client accuracy {mean_client_accuracy}")
            logging.write_logs("average_client_evaluation_after_local_training",
                               "accuracy " + str(mean_client_accuracy) + " loss " + str(mean_client_loss), "feddf")

            # Server side distillation feddf
            # 1- per-prototype predictions
            # 2- aggregation
            # 3- distillation

            stacked_client_preds = tf.stack(client_predictions)
            aggregated_predictions = tf.reduce_mean(stacked_client_preds, axis=0)
            # print("shape: ", tf.shape(aggregated_predictions))

            teacher_predictions = tf.data.Dataset.from_tensor_slices(tf.nn.softmax(aggregated_predictions, axis=1))

            img_pred = zip_img_and_pred(transfer_set_ds, teacher_predictions, batch_size_distill=batch_size_distill)

            server_distiller = distill(server_model, img_pred, distillation_epochs)
            server_weights = server_distiller.get_weights()
            server_model.set_weights(server_weights)

            print("[Server] Evaluation - Round: ", rnd)
            history = server_model.evaluate(test_ds, return_dict=True)
            with global_summary_writer.as_default():
                tf.summary.scalar('loss', tf.squeeze(history["loss"]), step=rnd)
                tf.summary.scalar('accuracy', tf.squeeze(history["accuracy"]), step=rnd)
            logging.write_logs("server_evaluation", history, "feddf")
