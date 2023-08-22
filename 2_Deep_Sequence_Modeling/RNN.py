import tensorflow as tf

my_rnn = RNN()
hidden_state = [0, 0, 0, 0]

sentence = ['I', 'love', 'recurrent', 'neural']

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction


# Similar to tf.keras.layers.SimpleRNN(rnn_units)
class RNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(RNNCell, self).__init__()

        # Initialize weights and biases
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([rnn_units, output_dim])

        # Initialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])

    def call(self, x):
        # Update hidden state
        self.h = tf.tanh(tf.matmul(self.h, self.W_hh) + tf.matmul(x, self.W_xh))

        # Compute output
        ouptut = tf.matmul(self.h, self.W_hy)

        # Return output and hidden state
        return output, self.h