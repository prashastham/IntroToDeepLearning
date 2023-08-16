import tensorflow as tf

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        # Initialize weights and biases
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward prop the inputs
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through non-linear activation
        output = tf.sigmoid(z)
        return output
    
# This class is already implemented in TF as
# layer = tf.keras.layers.Dense(units=n)