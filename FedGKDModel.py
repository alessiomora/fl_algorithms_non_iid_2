import tensorflow as tf


class FedGKDModel(tf.keras.Model):
    def __init__(self, model):
        super(FedGKDModel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, metrics, kd_loss, gamma=0.2):
        super(FedGKDModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.kd_loss = kd_loss
        self.gamma = gamma

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        xy, z = data
        x, y = xy

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ce_loss = self.compiled_loss(y, y_pred)
            # kl_divergence = tf.keras.losses.KLDivergence()
            kd_loss = self.kd_loss(tf.nn.softmax(z, axis=1), tf.nn.softmax(y_pred, axis=1))
            fedgkt_loss = ce_loss + (self.gamma/2) * kd_loss


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(fedgkt_loss, trainable_vars)
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
