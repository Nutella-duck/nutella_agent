# 학습하고 hyperopt 사용하는 데모

# 데이터 다운로드
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # 숫자 list 형태


# 데이터 변환
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension)이고 모든 원소가 0인 행렬
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. # results[i]에서 특정 인덱스의 위치를 1로 
    return results

x_train = vectorize_sequences(train_data) # 훈련 데이터를 벡터로 변환
x_test = vectorize_sequences(test_data) # 테스트 데이터를 벡터로 변환

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys

space = {'units': hp.uniform('units1', 64, 1024),
         'dropout': hp.uniform('dropout1', .25,.75),
         'activation': 'relu',
         'optimizer': hp.choice('optimizer', ['rmsprop', 'adadelta', 'adam']),
         'choice': hp.choice('num_layers', [{'layers':'two'}, {'layers':'three', 'units3': hp.uniform('units3', 64,1024), 'dropout3': hp.uniform('dropout3', 0.25, 0.75)}]),
         'epochs' : 3,
         'batch_size' : hp.choice('batch_size', [128, 512])
}

def objective(params):

  # 모델 설계
  from keras import models
  from keras import layers

  model = models.Sequential()
  model.add(layers.Dense(units=params['units'], activation=params['activation'], input_shape=(10000,)))
  model.add(layers.Dropout(params['dropout']))
  
  model.add(layers.Dense(units=params['units'], activation=params['activation']))
  model.add(layers.Dropout(params['dropout']))

  if params['choice']['layers']== 'three':
        model.add(layers.Dense(units=params['choice']['units3'], activation=params['activation'])) 
        model.add(layers.Dropout(params['choice']['dropout3'])) 

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

  return {'loss': -acc, 'status': STATUS_OK}

trials=Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)
print(best)
