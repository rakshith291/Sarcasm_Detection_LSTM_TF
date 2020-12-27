import tensorflow as tf

class Model :
    def __init__(self,vocab_size,max_len):
        self.vocab_size = vocab_size
        self.max_len = max_len
    def LSTM(self):
        embedding_dim = 16
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, embedding_dim, input_length=self.max_len),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
