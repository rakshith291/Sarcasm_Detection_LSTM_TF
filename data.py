import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Data :

    def __init__ (self,path):
        self.path = path
    def train_test(self):
        df = pd.read_json(self.path,lines=True)
        sentences = []
        labels = []
        for i,j in zip(df['headline'],df['is_sarcastic']):
            sentences.append(i)
            labels.append(j)

        training_size = round(len(sentences) * .75)
        training_sentences = sentences[0:training_size]
        testing_sentences = sentences[training_size:]
        training_labels = labels[0:training_size]
        testing_labels = labels[training_size:]

        vocab_size = 10000
        oov_tok = "<oov>"

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(training_sentences)

        max_length = 100
        trunc_type = 'post'
        padding_type = 'post'

        training_sequences = tokenizer.texts_to_sequences(training_sentences)
        training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type,
                                        truncating=trunc_type)
        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type,
                                       truncating=trunc_type)
        return training_padded, testing_padded,training_labels,testing_labels



