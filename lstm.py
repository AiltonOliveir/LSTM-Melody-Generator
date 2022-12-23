# -*- coding: utf-8 -*-
"""LSTM

"""

import sys, os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from music_decoder import play_chords
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras import optimizers

dataset_path = './Bach_chorales/'
train_path = dataset_path + 'train'
train_files = os.listdir(train_path)
df = pd.DataFrame()#append all files together
for csv_file in train_files:
  df_temp = pd.read_csv(train_path + '/' +csv_file)
  df = df.append(df_temp, ignore_index=True)
print('read all files')

train_data = df.values
df.head(10)

n_ts = 100
net_input = np.zeros((len(train_data)-n_ts,n_ts,4),dtype='uint8')
net_output =np.zeros((len(train_data)-n_ts,4),dtype='uint8')
for i in range(len(train_data)-n_ts):
  net_input[i,:,:] = train_data[i:i+n_ts]
  net_output[i,:] = train_data[i+n_ts]

print(net_input.shape)
print(net_output.shape)

opt = optimizers.Nadam(learning_rate=0.01)

model = Sequential()
model.add(LSTM(256,input_shape=(net_input.shape[1], net_input.shape[2]),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(net_input.shape[2]))
model.add(Activation('linear'))
model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])

model.summary()

filepath = "weights-lstm.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    
callbacks_list = [checkpoint]     
history = model.fit(net_input, net_output, epochs=3, batch_size=64,validation_split=0.3,callbacks=callbacks_list)

history.history

test_path = dataset_path + 'test'
test_files = os.listdir(test_path)
df = pd.DataFrame()#append all files together
for csv_file in test_files:
  df_temp = pd.read_csv(test_path + '/' +csv_file)
  df = df.append(df_temp, ignore_index=True)
print('read all files')
test_data = df.values
n_ts = 100
test_input = np.zeros((len(test_data)-n_ts,n_ts,4),dtype='uint8')
test_output =np.zeros((len(test_data)-n_ts,4),dtype='uint8')
for i in range(len(test_data)-n_ts):
  test_input[i,:,:] = test_data[i:i+n_ts]
  test_output[i,:] = test_data[i+n_ts]

accr = model.evaluate(test_input,test_output)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

prediction = model.predict(test_input)

prediction

prediction = prediction.astype('uint8')

print(prediction.shape)

"""#MIDI"""

play_chords(prediction[0:100])