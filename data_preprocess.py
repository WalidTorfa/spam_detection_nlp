import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def PreProcessing(file):
    data = pd.read_csv(file)
    email = []
    label = []
    for x in data["email"]:
        email.append(str(x))
    for l in data["label"]:
        label.append(l)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(email)
    seqs = tokenizer.texts_to_sequences(email)
    x_train, x_test, y_train, y_test = tts(seqs, label, test_size=0.3)
    maxlen = 100
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    return (x_test,x_train,y_train,y_test)
def preprocess_for_predict(sentence):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    seqs = tokenizer.texts_to_sequences(sentence)

    output = pad_sequences(seqs, maxlen=100)
    output = tf.convert_to_tensor(output)
    return output

