import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Loading the data
def load_and_prec(sample_size, 
        max_features,
        maxlen,
        data = pd.read_csv("content/pubmed_cr_hep_ctl_abstracts_clean.csv")):

    hep_pos = data[data['label_hep'] == 1].sample(sample_size)
    hep_neg = data[data['label_hep'] == 0].sample(sample_size)

    combined = pd.concat([hep_pos, hep_neg])

    X = combined['txt']
    y = combined['label_hep']

    print(len(X))
    hep_pos = combined[y == 1]
    hep_neg = combined[y == 0]
    print(len(hep_neg))
    print(len(hep_pos))

    print("X shape : ", X.shape)
    print("y shape : ", y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)  # .08 since the datasize is large enough.

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x_train))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_val = pad_sequences(x_val, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    return x_train, x_val, x_test, y_train, y_val, y_test, tokenizer.word_index