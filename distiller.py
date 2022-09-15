import tensorflow as tf


class Distiller(tf.keras.Model):
    def __init__(self, student):
        super(Distiller, self).__init__()
        self.student = student
        self.distillation_loss_fn = None
        # self.temperature = None

    def compile(
            self,
            optimizer,
            metrics,
            distillation_loss_fn,
            # temperature=1.0,
    ):
        """ Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        # self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = y

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            # Compute losses
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(student_predictions, axis=1),
                teacher_predictions
            )

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(distillation_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(
            tf.nn.softmax(student_predictions, axis=1), 
            teacher_predictions,
        )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Update the metrics.
        self.compiled_metrics.update_state(
            tf.nn.softmax(y_prediction, axis=1),
            y,
        )

        return {m.name: m.result() for m in self.metrics}

    # implement the call method
    def call(self, inputs):
        return self.student(inputs)