'''
Local-Global Imitation Correction
'''

import tensorflow as tf

class FedLGICModel(tf.keras.Model):
    def __init__(self, local_model_mlb, global_model_mlb):
        super(FedLGICModel, self).__init__()
        # both local_model mlb_model are instance of custom mlb model
        self.model = local_model_mlb
        self.global_model = global_model_mlb

    def compile(self, optimizer, loss, metrics, kd_loss=tf.keras.losses.MeanSquaredError(), lambda_1=0.5, lambda_2=0.5):
        super(FedLGICModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.kd_loss = kd_loss
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            out_of_global = self.global_model(x, return_feature=True)
            out_of_local = self.model(x, return_feature=True)

            local_features = out_of_local[:-1]
            global_features = out_of_global[:-1]

            local_log_probs = out_of_local[-1]
            # global_log_probs = out_of_global[-1]

            correction_losses = []
            imitation_losses = []
            num_blocks = len(local_features)

            # usual cross-entropy
            ce_loss = self.compiled_loss(y, local_log_probs, regularization_losses=self.losses)
            ## Compute loss from hybrid branches
            for it in range(num_blocks):
                c_l = self.model(global_features[it], return_feature=False, level=it+1)
                cl_loss = self.kd_loss(global_features[it + 1], c_l)
                correction_losses.append(cl_loss)

                i_l = self.global_model(local_features[it], return_feature=False, level=it+1)
                il_loss = self.kd_loss(local_features[it + 1], i_l)
                imitation_losses.append(il_loss)
                # this_log_prob = self.global_model(local_features[it], level=it + 1)
                # this_ce = self.compiled_loss(y, this_log_prob)
                # this_kl = self.kd_loss(tf.nn.softmax(local_log_probs, axis=1), tf.nn.softmax(this_log_prob, axis=1))
                # ce_branch.append(this_ce)
                # kl_branch.append(this_kl)

            # tf.print("shape ", tf.stack(this_ce))
            correction_loss = tf.reduce_mean(tf.stack(correction_losses))
            imitation_loss = tf.reduce_mean(tf.stack(imitation_losses))
            # kd_loss = tf.reduce_mean(tf.stack(kl_branch))
            fedlgic_loss = ce_loss + self.lambda_1 * correction_loss + self.lambda_2 * imitation_loss

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(fedlgic_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, local_log_probs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        outputs = self.model(x, return_feature=True)  # Forward pass
        self.compiled_loss(y, outputs[-1], regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, outputs[-1])
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.model.get_weights()

    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [loss_tracker]
