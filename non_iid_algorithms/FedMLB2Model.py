import tensorflow as tf


class FedMLB2Model(tf.keras.Model):
    """ FedMLB implementation from the paper https://arxiv.org/abs/2207.06936
    Based on the original implementation at https://github.com/jinkyu032/FedMLB. """

    def __init__(self, local_model_mlb, global_model_mlb):
        super(FedMLB2Model, self).__init__()
        # both local_model mlb_model are instance of custom mlb model
        self.model = local_model_mlb
        self.global_model = global_model_mlb

    def compile(self, optimizer, loss, metrics, kd_loss, lambda_1=0.1, lambda_2=0.3):
        super(FedMLB2Model, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.kd_loss = kd_loss
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            out_of_local = self.model(x, return_feature=True)

            local_features = out_of_local[:-1]

            log_probs = out_of_local[-1]

            ce_branch = []
            kl_branch = []
            num_branch = len(local_features)

            ce_loss = self.compiled_loss(y, log_probs, regularization_losses=self.losses)
            ## Compute loss from hybrid branches
            for it in range(num_branch):
                # tf.print("it ", it)
                this_log_prob = self.global_model(local_features[it], level=it + 1)
                this_ce = self.compiled_loss(y, this_log_prob)
                # kd_loss(y_true, y_pred)
                # this_kl = self.kd_loss(tf.nn.softmax(log_probs, axis=1), tf.nn.softmax(this_log_prob, axis=1))
                this_kl = self.kd_loss(tf.nn.softmax(this_log_prob, axis=1), tf.nn.softmax(log_probs, axis=1))
                ce_branch.append(this_ce)
                kl_branch.append(this_kl)

            # tf.print("shape ", tf.stack(this_ce))
            ce_hybrid_loss = tf.reduce_mean(tf.stack(ce_branch))
            kd_loss = tf.reduce_mean(tf.stack(kl_branch))
            fedmlb_loss = ce_loss + self.lambda_1 * ce_hybrid_loss + self.lambda_2 * kd_loss

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(fedmlb_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, log_probs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_pred = self.model(x)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.model.get_weights()
