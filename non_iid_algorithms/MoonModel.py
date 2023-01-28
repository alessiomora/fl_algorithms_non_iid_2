import tensorflow as tf


class MoonModel(tf.keras.Model):
    """ MOON implementation from the paper and their github"""

    def __init__(self, model, last_model, global_model):
        super(MoonModel, self).__init__()
        self.model = model
        self.last_model = last_model
        self.global_model = global_model

    def compile(self, optimizer, loss, metrics, mu=1.0, temperature=0.5):
        super(MoonModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.mu = mu
        self.t = temperature

    def train_step(self, data):
        x, y = data
        cosine_sim = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
        ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            pro1, y_pred = self.model(x, training=True)  # Forward pass
            pro2, _ = self.global_model(x, training=True)
            pro3, _ = self.last_model(x, training=True)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_ce = self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)

            logits_list = []

            positive = cosine_sim(pro2, pro1)
            logits_list.append(tf.reshape(positive, [-1, 1]))

            negative = cosine_sim(pro3, pro1)
            logits_list.append(tf.reshape(negative, [-1, 1]))

            logits = tf.concat(logits_list, axis=1)
            logits = logits/self.t

            labels = tf.zeros(tf.shape(y))
            loss_contrastive = self.mu * ce(labels, logits)
            moon_loss = loss_ce + loss_contrastive

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(moon_loss, trainable_vars)
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


