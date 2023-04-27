import tensorflow as tf

class FedMAXModel(tf.keras.Model):
    """ FedMax implementation from https://github.com/weichennone/FedMAX/blob/91fa2a8ce5ba2ccaf2594d12cf11874c2f34d387/digit_object_recognition/models/Update.py#L47 """

    def __init__(self, model):
        super(FedMAXModel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, metrics, kd_loss, beta=1000):
        super(FedMAXModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.beta = beta
        self.kd_loss = kd_loss

    def train_step(self, data):
        x, y = data

        # zero_mat = torch.zeros(out.size()).cuda()
        # softmax = nn.Softmax(dim=1)
        # logsoftmax = nn.LogSoftmax(dim=1)
        #
        # kldiv = nn.KLDivLoss(reduce=True)
        # cost = beta * kldiv(logsoftmax(out), softmax(zero_mat))

        with tf.GradientTape() as tape:
            act, y_pred = self.model(x, training=True)  # Forward pass
            zero_mat = tf.zeros(tf.shape(act))
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_ce = self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)

            loss_max = self.beta * self.kd_loss(tf.nn.softmax(zero_mat, axis=1), tf.nn.softmax(act, axis=1))
            loss_fedmax = loss_ce + loss_max

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss_fedmax, trainable_vars)
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

        _, y_pred = self.model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}
