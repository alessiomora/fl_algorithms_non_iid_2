import tensorflow as tf
import numpy as np


class FedNTDModel(tf.keras.Model):
    """ FedNTD implementation from the paper https://arxiv.org/abs/2106.03097 """

    def __init__(self, model):
        super(FedNTDModel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, metrics, kd_loss, temperature=1.0, beta=0.3):
        super(FedNTDModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.kd_loss = kd_loss
        self.beta = beta
        self.t = temperature

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        xy, z = data
        x, y = xy

        # shape = tf.shape(tf.add(z, 1))
        # tf.print(shape)
        # discarding true-class logits
        del_idx_2 = y
        # shape = z.shape
        # print(shape)
        # mask = np.ones((shape[0], shape[1]), dtype=bool)
        ones = tf.ones((tf.shape(z)[0], tf.shape(z)[1]), dtype=tf.bool)
        # tf.print(ones)
        rows = tf.range(tf.shape(z)[0])
        tf.print(rows)
        # tf.print(tf.shape(del_ind)[0])
        tf.print(tf.shape(del_idx_2))
        to_delete = tf.squeeze(tf.cast(del_idx_2, tf.int32), axis=1)

        tf.print(to_delete)
        del_idx = tf.stack([rows, to_delete], axis=1)
        # tf.print(del_idx)
        updates = tf.fill([tf.shape(z)[0]], False)
        mask = tf.tensor_scatter_nd_update(ones, del_idx, updates)
        # tf.print(mask)
        # mask[range(2), del_ind] = False
        masked = tf.boolean_mask(z, mask)
        # tf.print(tf.shape(masked), summarize=-1)
        z_ntd = tf.reshape(masked, (tf.shape(z)[0], tf.shape(z)[1] - 1))
        # tf.print(tf.shape(z_ntd))

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            # tf.print(tf.shape(y_pred))
            # discarding true-class logits
            # del_ind = y
            # shape = tf.shape(y_pred)
            # mask = np.ones((shape[0], shape[1]), dtype=bool)
            # mask[range(2), del_ind] = False
            # masked = tf.boolean_mask(y_pred, mask)
            # y_pred_ntd = tf.reshape(masked, (shape[0], shape[1] - 1))
            # print(y_pred_ntd)

            # ---
            # shape = tf.shape(tf.add(z, 1))
            # tf.print(shape)
            # discarding true-class logits
            del_ind = y
            ones = tf.ones((tf.shape(z)[0], tf.shape(z)[1]), dtype=tf.bool)
            rows = tf.range(tf.shape(z)[0])
            del_idx = tf.stack([rows, tf.squeeze(tf.cast(del_ind, tf.int32), axis=1)], axis=1)
            updates = tf.fill([tf.shape(z)[0]], False)
            mask = tf.tensor_scatter_nd_update(ones, del_idx, updates)
            masked = tf.boolean_mask(y_pred, mask)
            y_pred_ntd = tf.reshape(masked, (tf.shape(z)[0], tf.shape(z)[1] - 1))

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ce_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # remove elements corresponding to true classe
            kd_loss = self.kd_loss(tf.nn.softmax(y_pred_ntd / self.t, axis=1),
                                   tf.nn.softmax(z_ntd / self.t, axis=1))
            fedntd_loss = ce_loss + self.beta * kd_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(fedntd_loss, trainable_vars)
        # gradients = tape.gradient(ce_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_pred = self.model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [loss_tracker]
