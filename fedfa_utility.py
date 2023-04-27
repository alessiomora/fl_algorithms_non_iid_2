import tensorflow as tf


# utility function for FedFA's statistics aggregation
def aggregate_clients_var(values):
    """
    Return the variance of the list values passed as input.

    Parameters:

        values: list of numbers

    Returns:
        var: variance as a tf.float
    """
    tmp = tf.stack(values)
    var = tf.math.reduce_variance(tmp)
    return var
