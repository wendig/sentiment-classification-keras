'''Train a recurrent convolutional network on the IMDB sentiment
classification task.
Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, AtrousConvolution1D
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
import numpy as np
# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
(x_train, y_train_tmp), (x_test, y_test_tmp) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


print('one-hot encoding')
#one-hot encoding, numpy only
#improves test accuracy, 0.849 -> 0.854
y_train = np.zeros((len(y_train_tmp),2))
y_test  = np.zeros((len(y_test_tmp),2))

y_train[np.arange(len(y_train_tmp)),y_train_tmp] = 1
y_test[np.arange(len(y_test_tmp)),y_test_tmp] = 1



print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.2))
model.add(AtrousConvolution1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size,return_sequences=True))

model.add(Dropout(0.6))
model.add(AtrousConvolution1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))


model.add(Dense(2, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
        
#==============================================================================
# 2. model
# model.add(Embedding(max_features, 128))
# model.add(Convolution1D(64, 3, border_mode='same'))
# model.add(Convolution1D(32, 3, border_mode='same'))
# model.add(Convolution1D(16, 3, border_mode='same'))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(180,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 
#==============================================================================
