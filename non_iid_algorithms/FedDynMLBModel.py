import tensorflow as tf


class FedDynMLBModel(tf.keras.Model):
    """ FedDyn implementation from the paper and their github"""

    def __init__(self, local_model_mlb, global_model_mlb):
        super(FedDynMLBModel, self).__init__()
        # both local_model mlb_model are instance of custom mlb model
        self.model = local_model_mlb
        self.global_model = global_model_mlb

    def compile(self, optimizer, loss, metrics, kd_loss, avg_mdl_param, local_grad_vector, alpha=0.1, lambda_1=0.1, lambda_2=0.3):
        super(FedDynMLBModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # DYN
        self.alpha = alpha
        avg_mdl_param_concat = tf.concat([tf.reshape(ww, [-1]) for ww in avg_mdl_param], axis=0)
        local_grad_vector_concat = tf.concat([tf.reshape(ww, [-1]) for ww in local_grad_vector], axis=0)
        self.local_grad_vector_minus_avg_mdl_param = - avg_mdl_param_concat + local_grad_vector_concat

        # MLB
        self.kd_loss = kd_loss
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            out_of_local = self.model(x, return_feature=True)

            local_features = out_of_local[:-1]

            logits = out_of_local[-1]

            ce_branch = []
            kl_branch = []
            num_branch = len(local_features)

            loss_ce = self.compiled_loss(y, logits, regularization_losses=self.model.losses)

            ## Compute loss from hybrid branches
            for it in range(num_branch):
                # tf.print("it ", it)
                this_logits = self.global_model(local_features[it], level=it + 1)
                this_ce = self.cross_entropy(y, this_logits)
                this_kl = self.kd_loss(tf.nn.softmax(this_logits), tf.nn.softmax(logits))
                ce_branch.append(this_ce)
                kl_branch.append(this_kl)

            ce_hybrid_loss = tf.reduce_mean(tf.stack(ce_branch))
            kd_loss = tf.reduce_mean(tf.stack(kl_branch))

            local_par_list = tf.concat([tf.reshape(ww, [-1]) for ww in self.model.trainable_variables], axis=0)
            loss_dyn = self.alpha * tf.reduce_sum(local_par_list * self.local_grad_vector_minus_avg_mdl_param)
            loss_mlb = self.lambda_1 * ce_hybrid_loss + self.lambda_2 * kd_loss

            loss = loss_dyn + loss_mlb + loss_ce

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.model.get_weights()

    def get_global_weights(self):
        return self.global_model.get_weights()


