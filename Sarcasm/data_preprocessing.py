import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
def preprocessing(data):
    data = pd.read_csv(data)
    sentence = []
    label = []
    for s in data['headline']:
        sentence.append(s)
    for l in data['is_sarcastic']:
        label.append(l)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    seq = tokenizer.texts_to_sequences(sentence)
    seq = pad_sequences(seq, maxlen=100)
    x_train, x_test, y_train, y_test = tts(seq, label, test_size=0.1)
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)
    x_test = tf.convert_to_tensor(x_test)
    return(x_train,y_train,x_test,y_test)
def predict(input):
    model = load_model("Sarcasm.h5")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input)
    seqs1 = tokenizer.texts_to_sequences(input)
    output1 = pad_sequences(seqs1, maxlen=100)
    output1 = tf.convert_to_tensor(output1)
    output1 = model.predict(output1)
    if output1 >= 0.5:
        print('Sarcasm')
    else:
        print('Not Sarcasm')
    print(output1)

