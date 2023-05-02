import tensorflow as tf

class DMFLodel(tf.keras.Model):

    def __init__(self, model):
        super(DMFLodel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, metrics, class_occ, tau):
        super(DMFLodel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        s = tf.constant([1e-8], tf.float32)
        class_occ_tensor = tf.constant(class_occ, tf.float32)
        # To not have zeros and generate nan
        self.class_occ = tf.math.maximum(class_occ_tensor, s)
        self.tau = tau

    def train_step(self, data):
        x, y = data

        def logit_calibrated_loss(y, logit, class_occ, tau):
            # tf.print("\n distrib ", class_occ, summarize=-1)
            cal_logit = tf.math.exp(
                logit - tf.repeat(
                    tf.expand_dims(tau * tf.math.pow(class_occ, -1 / 4), axis=0),
                    repeats=tf.shape(y)[0],
                    axis=0)
            )
            # y_logit = tf.gather(cal_logit, indices=y)
            # i_not_equal_y = tf.gather(cal_logit, indices=tf.expand_dims(y, axis=-1))
            y_logit = tf.gather(cal_logit, indices=y, axis=1, batch_dims=1)

            # extract true class logit needed in the denominator
            z = logit
            del_ind = y
            ones = tf.ones((tf.shape(z)[0], tf.shape(z)[1]), dtype=tf.bool)
            rows = tf.range(tf.shape(z)[0])
            del_idx = tf.stack([rows, tf.squeeze(tf.cast(del_ind, tf.int32), axis=1)], axis=1)
            updates = tf.fill([tf.shape(z)[0]], False)
            mask = tf.tensor_scatter_nd_update(ones, del_idx, updates)
            masked = tf.boolean_mask(z, mask)
            z_ntd = tf.reshape(masked, (tf.shape(z)[0], tf.shape(z)[1] - 1))

            denominator = y_logit + tf.reduce_sum(z_ntd, axis=-1, keepdims=True)
            calibrated_loss = -tf.math.log(tf.math.divide_no_nan(y_logit, denominator))

            loss = tf.reduce_mean(calibrated_loss)
            tf.print(loss)
            return loss
            # return logit


        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            loss_ce = logit_calibrated_loss(y, y_pred, self.class_occ, self.tau)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss_ce, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        result = {m.name: m.result() for m in self.metrics}
        return result

