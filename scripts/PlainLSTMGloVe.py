'''
Created on 27.01.2017

@author: bastianbertram

'''
import numpy as np
np.random.seed(123)

import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import util.Metrics as met

MAX_LENGTH = 140

INPUT_DIM = 20000
OUTPUT_DIM = 100

INPUT_LENGTH = 25

BATCH_SIZE = 100
NB_EPOCH = 20
CALLBACK_PATH = "../data/callback/PlainLSTMGloVe/weights-improvement-{epoch:02d}-{val_acc:.2f}.txt"

TRAINING_FILE = '../data/normalized/twitter-2016devCOMPLETE-outputNORMALIZED2.txt'
TEST_FILE = '../data/normalized/twitter-2016test-outputNORMALIZED2.txt'

PRED_FILE = 'PlainLSTMGloVe/y_pred.txt'
TRUE_FILE = 'PlainLSTMGloVe/y_true.txt'




token = Tokenizer()

print("Currently loading data...")
names = ['id','topic','yLabel','tweet']

train_data = pd.read_table(TRAINING_FILE, sep="\t", header=None, names=names)
y_train = to_categorical(train_data['yLabel'].values, nb_classes = 5)
x_train = np.array(train_data["tweet"])

test_data = pd.read_table(TEST_FILE, sep="\t", header=None, names=names)
y_test = to_categorical(test_data["yLabel"].values, nb_classes=5)
x_test = test_data["tweet"].values
print("Finished loading data...")




print("Start padding...")
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)

token.fit_on_texts(x_test)
x_test = token.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=INPUT_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=INPUT_LENGTH)
print("Finished padding...")





print("Prepare Embedding Layer...")
word_index = token.word_index

f = open("../data/glove.6B.100d.txt", encoding="utf-8")
embedding_index = {}
for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:])
    embedding_index[word] = coefs
f.close()


#print("Found %s word vectors." % len(embedding_index))

nb_words = len(word_index)
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, index in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        





print("Building model...")

model = Sequential()
model.add(Embedding(input_dim=nb_words+1, output_dim=OUTPUT_DIM, weights = [embedding_matrix], input_length=INPUT_LENGTH, trainable=False))
model.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='valid', activation='relu', subsample_length=1))
model.add(MaxPooling1D(pool_length=4))
model.add(LSTM(70))
model.add(Dense(5, activation = "softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''
model = Sequential()
model.add(Embedding(input_dim=nb_words+1, output_dim=OUTPUT_DIM, weights=[embedding_matrix], input_length=INPUT_LENGTH, trainable=False))
model.add(LSTM(25, return_sequences=True, batch_input_shape=(None, INPUT_LENGTH), dropout_W=0.2, dropout_U=0.2))
model.add(LSTM(30))
model.add(Dense(5, activation='softmax'))
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=['accuracy'])'''


print('Train...')
checkpoint = ModelCheckpoint(CALLBACK_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=callbacks_list, 
          validation_data=(x_test, y_test))
score, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

y_pred = model.predict_classes(x=x_test, batch_size=BATCH_SIZE, verbose=1)
met.eval_MAE(test_data=test_data, y_pred=y_pred, PRED_FILE=PRED_FILE, TRUE_FILE=TRUE_FILE)


print('Test score:', score)
print('Test accuracy:', accuracy)


