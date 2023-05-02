import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

path = os.path.join("cifar100_dirichlet", "0.1", "distribution.npy")
smpls_loaded = np.load(path)


def generate_dirichlet_samples(num_of_classes, alpha, num_of_clients,
                               num_of_examples_per_label):
    """Generate samples from a dirichlet distribution based on alpha parameter.
    Samples will have the shape (num_of_clients, num_of_classes).
    Returns an int tensor with shape (num_of_clients, num_of_classes)."""
    for _ in range(0, 10):
        alpha_tensor = tf.fill(num_of_classes, alpha)
        # alpha_tensor = alpha * prior_distrib
        # print(alpha_tensor)
        dist = tfp.distributions.Dirichlet(tf.cast(alpha_tensor, tf.float32))
        samples = dist.sample(num_of_clients)
        # Cast to integer for an integer number of examples per label per client
        int_samples_transpose = tf.cast(tf.round(samples * num_of_examples_per_label), tf.int32)
        # int_samples_transpose = tf.transpose(int_samples, [1, 0])
        # print("reduce_sum", tf.reduce_sum(int_samples_transpose, axis=1))
        correctly_generated = tf.reduce_min(tf.reduce_sum(int_samples_transpose, axis=1))
        if tf.cast(correctly_generated, tf.float32) != tf.constant(0.0, tf.float32):
            break
        print("Generated some clients without any examples. Retrying..")

    return int_samples_transpose


print(smpls_loaded)
print("Sum of examples per client")
print(tf.reduce_sum(smpls_loaded, axis=1))
print("Sum of examples per label")
print(tf.reduce_sum(smpls_loaded, axis=0))
#
# smpls_loaded = generate_dirichlet_samples(num_of_classes=10, alpha=0.1, num_of_clients=20,
#                                num_of_examples_per_label=2500)
#
# print(tf.reduce_sum(smpls_loaded, axis=1))
# print(tf.reduce_sum(smpls_loaded, axis=0))
