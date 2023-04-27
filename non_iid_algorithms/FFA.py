import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class FFALayer(keras.layers.Layer):
    def __init__(self, prob=0.5, eps=1e-6, momentum1=0.99, momentum2=0.99, seed=None, nfeat=None):
        super(FFALayer, self).__init__()
        self.prob = prob
        self.eps = eps
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.seed = seed
        self.running_var_mean_bmic = tf.ones(nfeat,)  # BROADCAST BY THE SERVER
        self.running_var_std_bmic = tf.ones(nfeat,)
        self.running_mean_bmic = tf.Variable(tf.zeros(nfeat,), trainable=False)  # TO BE UPLOADED
        self.running_std_bmic = tf.Variable(tf.ones(nfeat,), trainable=False)

        # self.nfeat = nfeat

    # def build(self, input_shape):
    #     self.running_var_mean_bmic = tf.ones((int(input_shape[-1]),)) # BROADCAST BY THE SERVER
    #     self.running_var_std_bmic = tf.ones((int(input_shape[-1]),))
    #     self.running_mean_bmic = tf.Variable(tf.zeros((int(input_shape[-1]),)), trainable=False) # TO BE UPLOADED
    #     self.running_std_bmic = tf.Variable(tf.ones((int(input_shape[-1]),)), trainable=False)

    @tf.function
    def call(self, x, training=None):
        if not training:
            return x
        if tf.random.uniform([]) > self.prob:
            return x

        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        std = tf.math.reduce_variance(x, axis=[1, 2], keepdims=False) + self.eps
        std = tf.math.sqrt(std)

        # to upload
        self.running_mean_bmic.assign(self.running_mean_bmic * self.momentum1 +
                                      tf.reduce_mean(mean, axis=0, keepdims=False) * (1 - self.momentum1))
        self.running_std_bmic.assign(self.running_std_bmic * self.momentum1 +
                                     tf.reduce_mean(std, axis=0, keepdims=False) * (1 - self.momentum1))

        var_mu = var(mean, eps=self.eps)
        var_std = var(std, eps=self.eps)

        # IO: input[None, 32, 32, 3]
        # LORO: input[None, 3, 32, 32]
        running_var_mean_bmic = 1 / (1 + 1 / (self.running_var_mean_bmic + self.eps)) # downloaded from the server
        # gamma_mu = x.shape[1] * running_var_mean_bmic / sum(running_var_mean_bmic)
        gamma_mu = tf.cast(tf.shape(x)[3], tf.float32) * running_var_mean_bmic / tf.reduce_sum(running_var_mean_bmic)

        running_var_std_bmic = 1 / (1 + 1 / (self.running_var_std_bmic + self.eps)) # downloaded from the server
        # gamma_std = x.shape[1] * running_var_std_bmic / sum(running_var_std_bmic)
        gamma_std = tf.cast(tf.shape(x)[3], tf.float32) * running_var_std_bmic / tf.reduce_sum(running_var_std_bmic)

        var_mu = (gamma_mu + 1) * var_mu
        var_std = (gamma_std + 1) * var_std

        # var_mu = var_mu.sqrt().repeat(x.shape[0], 1)
        var_mu = tf.experimental.numpy.tile(tf.math.sqrt(var_mu), (tf.shape(x)[0], 1))
        # var_std = var_std.sqrt().repeat(x.shape[0], 1)
        var_std = tf.experimental.numpy.tile(tf.math.sqrt(var_std), (tf.shape(x)[0], 1))

        beta = gaussian_sampling(mean, var_mu, self.seed)
        gamma = gaussian_sampling(std, var_std, self.seed)

        # x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = (x - tf.reshape(mean, (tf.shape(x)[0], 1, 1, tf.shape(x)[3]))) / tf.reshape(std, (
            tf.shape(x)[0], 1, 1, tf.shape(x)[3]))

        # x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * tf.reshape(gamma, (tf.shape(x)[0], 1, 1, tf.shape(x)[3])) + tf.reshape(beta, (
            tf.shape(x)[0], 1, 1, tf.shape(x)[3]))

        # tf.print("-----------------", tf.shape(x))
        return x


def gaussian_sampling(mu, std, seed):
    e = tf.random.normal(tf.shape(std), seed=seed)
    # e = torch.randn_like(std)
    tmp = tf.math.multiply(e, std)
    # z = e.mul(std).add_(mu)
    return tf.math.add(tmp, mu)


def var(x, eps):
    # t = x.var(dim=0, keepdim=False) + self.eps
    t = tf.math.reduce_variance(input_tensor=x, axis=0, keepdims=True) + eps
    return t

# usage
# layer = FFA(nfeat=3)
#
# targets = tf.random.uniform([3, 3, 3, 3])
# print(targets)
# y = layer(targets, training=True)
# print(y)
