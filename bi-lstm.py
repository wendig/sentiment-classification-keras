
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, BatchNormalization, Dropout
from keras.layers import LSTM, Bidirectional
from keras.datasets import imdb
import numpy as np

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

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
y_train = np.zeros((len(y_train_tmp),2))
y_test  = np.zeros((len(y_test_tmp),2))

y_train[np.arange(len(y_train_tmp)),y_train_tmp] = 1
y_test[np.arange(len(y_test_tmp)),y_test_tmp] = 1


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(BatchNormalization())#maybe
model.add(Bidirectional(LSTM(128, dropout=0.2, 
                             recurrent_dropout=0.2, 
                             return_sequences=True)))
model.add(Bidirectional(LSTM(32)))#remove for faster training()
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)