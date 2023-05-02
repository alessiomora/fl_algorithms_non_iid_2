import numpy as np
import tensorflow as tf
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

path = os.path.join("cifar100_dirichlet_train_and_test", "0.3", "distribution_train.npy")
smpls_loaded = np.load(path)

print(smpls_loaded)
print("Sum of examples per client")
print(tf.reduce_sum(smpls_loaded, axis=1))
print("Sum of examples per label")
print(tf.reduce_sum(smpls_loaded, axis=0))