from data import Data
from model import Model
import numpy as np

def main():

    data_  = Data ('archive/Sarcasm_Headlines_Dataset.json')
    train_padded,test_padded ,training_labels,testing_labels = data_.train_test()
    training_padded = np.array(train_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(test_padded)
    testing_labels = np.array(testing_labels)

    m = Model(10000,100)
    model = m.LSTM()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_padded,training_labels,validation_data=(testing_padded,testing_labels),epochs=30,verbose=2)




if __name__ == '__main__':
    main()


