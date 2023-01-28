import tensorflow as tf


class FedDynModel(tf.keras.Model):
    """ FedDyn implementation from the paper and their github"""

    def __init__(self, model):
        super(FedDynModel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, metrics, avg_mdl_param, local_grad_vector, alpha=0.1):
        super(FedDynModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.alpha = alpha
        # self.avg_mdl_param = tf.concat([tf.reshape(ww, [-1]) for ww in avg_mdl_param], axis=0)
        avg_mdl_param_concat = tf.concat([tf.reshape(ww, [-1]) for ww in avg_mdl_param], axis=0)
        # self.local_grad_vector = tf.concat([tf.reshape(ww, [-1]) for ww in local_grad_vector], axis=0)
        local_grad_vector_concat = tf.concat([tf.reshape(ww, [-1]) for ww in local_grad_vector], axis=0)
        self.local_grad_vector_minus_avg_mdl_param = - avg_mdl_param_concat + local_grad_vector_concat

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_ce = self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)

            local_par_list = tf.concat([tf.reshape(ww, [-1]) for ww in self.model.trainable_variables], axis=0)
            # loss_dyn = self.alpha * tf.reduce_sum(local_par_list * (-self.avg_mdl_param + self.local_grad_vector))
            loss_dyn = self.alpha * tf.reduce_sum(local_par_list * self.local_grad_vector_minus_avg_mdl_param)
            feddyn_loss = loss_ce + loss_dyn

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(feddyn_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # print(tf.shape(y_pred))
        result = {m.name: m.result() for m in self.metrics}
        # result.update({"loss_ce": self.compiled_loss(y, y_pred), "decay": loss_ce, "loss algo": loss_dyn})
        return result

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.model.get_weights()


