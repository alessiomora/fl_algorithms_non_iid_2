import tensorflow as tf


class FedProxModel(tf.keras.Model):
    """ FedProx implementation from the paper https://arxiv.org/abs/1812.06127 """

    def __init__(self, model):
        super(FedProxModel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, metrics, initial_weights, mu=0.01):
        super(FedProxModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.mu = mu
        self.initial_weights = initial_weights

    def train_step(self, data):
        def difference_model_norm_2_square(global_model, local_model):
            """Calculates the squared l2 norm of a model difference (i.e.
            local_model - global_model)
            Args:
                global_model: the model broadcast by the server
                local_model: the current, in-training model

            Returns: the squared norm

            """
            model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                                     local_model,
                                                     global_model)
            squared_norm = tf.square(tf.linalg.global_norm(model_difference))
            return squared_norm

        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            prox_term = (self.mu / 2) * difference_model_norm_2_square(
                self.model.weights, self.initial_weights)
            fedprox_loss = loss + prox_term

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(fedprox_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # tf.print(fedprox_loss)
        result = {m.name: m.result() for m in self.metrics}
        result.update({"prox_term_loss": fedprox_loss})
        return result

    def test_step(self, data):
        x, y = data

        y_pred = self.model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}
