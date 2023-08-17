import tensorflow as tf

lr = 1

def compute_loss(weights):
    #return predicted weights - real weights
    return tf.reduce_mean(tf.pow(weights - real_weights, 2))


def gradient_descent():
  # Code to implement gradient descent
    weights = tf.Variable([tf.random.normal()])

    while True:
        with tf.GradientTape() as g:
            loss = compute_loss(weights)
            gradient = g.gradient(loss, weights)

        weights = weights - lr * gradient