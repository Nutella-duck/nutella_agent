# 데이터 다운로드
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 데이터 변환
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. 
    return results

x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data) 
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from nutellaAgent import hpo
from nutellaAgent import nu_simple_fmin
from sklearn.metrics import roc_auc_score
import sys

def objective(params):

  # 모델 설계
  from keras import models
  from keras import layers

  model = models.Sequential()
  model.add(layers.Dense(units=params['units'], activation='relu', input_shape=(10000,)))
  model.add(layers.Dropout(params['dropout']))
  
  model.add(layers.Dense(units=params['units'], activation='relu'))
  model.add(layers.Dropout(params['dropout']))

  model.add(layers.Dense(1, activation='sigmoid'))

  # 컴파일
  model.compile(optimizer=params['optimizer'],
                loss='binary_crossentropy',
                metrics=['acc'])

  # data 설정
  x_val = x_train[:10000]
  partial_x_train = x_train[10000:]
  y_val = y_train[:10000]
  partial_y_train = y_train[10000:]

  # 학습
  history = model.fit(partial_x_train,
                      partial_y_train,
                      epochs=params['epochs'],
                      batch_size=params['batch_size'],
                      validation_data=(x_val, y_val))
  
  loss, acc = model.evaluate(x_test, y_test)

  return {'loss': loss, 'status': hpo.STATUS_OK}

best, trials = nu_simple_fmin("", objective)
