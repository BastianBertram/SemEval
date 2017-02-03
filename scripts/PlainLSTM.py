'''
Created on 25.01.2017

@author: bastianbertram
'''
import numpy as np
np.random.seed(123)

import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import util.Metrics as met

MAX_LENGTH = 140

INPUT_DIM = 20000
OUTPUT_DIM = 300

INPUT_LENGTH = 25

BATCH_SIZE = 100
NB_EPOCH = 1

CALLBACK_PATH = "../data/callback/PlainLSTM/weights-improvement-{epoch:02d}-{val_acc:.2f}.txt"

TRAINING_FILE = '../data/normalized/twitter-2016devCOMPLETE-outputNORMALIZED2.txt'
TEST_FILE = '../data/normalized/twitter-2016test-outputNORMALIZED2.txt'

PRED_FILE = 'PlainLSTM/y_pred.txt'
TRUE_FILE = 'PlainLSTM/y_true.txt'


print("Currently loading data...")

names = ['id','topic','yLabel','tweet']

train_data = pd.read_table(TRAINING_FILE, sep="\t", header=None, names=names)
x_train = np.array(train_data["tweet"])
y_train = to_categorical(train_data['yLabel'].values, nb_classes = 5)

for i in range(0, len(x_train)):
    x_train[i] = one_hot(x_train[i], n=20000, split=" ")

test_data = pd.read_table(TEST_FILE, sep="\t", header=None, names=names)
x_test = test_data["tweet"].values
y_test = to_categorical(test_data["yLabel"].values, nb_classes=5)
for i in range(0, len(x_test)):
    x_test[i] = one_hot(x_test[i], n=20000, split=" ")
    



print("Finished loading data...")

print("Start padding...")
x_train = sequence.pad_sequences(x_train, maxlen=33)
x_test = sequence.pad_sequences(x_test, maxlen=33)
print("Finished padding...")

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("Training shape: ", x_train.shape[1])




print("Building model...")
'''
model = Sequential()
model.add(Embedding(20000, 100, input_length=33, dropout=0.25))
model.add(Convolution1D(nb_filter=64,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=4))
model.add(LSTM(70))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''


model = Sequential()
model.add(Embedding(INPUT_DIM, OUTPUT_DIM, input_length=INPUT_LENGTH, dropout=0.2))
model.add(LSTM(60, return_sequences=True, batch_input_shape=(None, INPUT_LENGTH), dropout_W=0.2, dropout_U=0.2))
model.add(LSTM(80))
model.add(Dense(5, activation='softmax'))
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=['accuracy'])



print('Train...')
checkpoint = ModelCheckpoint(CALLBACK_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=callbacks_list,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)


y_pred = model.predict_classes(x=x_test, batch_size=BATCH_SIZE, verbose=1)
met.eval_MAE(test_data=test_data, y_pred=y_pred, PRED_FILE=PRED_FILE, TRUE_FILE=TRUE_FILE)

print('Test score:', score)
print('Test accuracy:', acc)


