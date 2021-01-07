from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from nutellaAgent import nu_fmin, hpo

max_features = 20000
maxlen = 80 

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

def objective(params):
    model = Sequential()
    model.add(Embedding(max_features, units=params['units']))
    model.add(LSTM(units=params['units'], dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout']))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    space = {'units': hpo.hp.uniform('units', 64, 1024),
            'dropout': hpo.hp.uniform('dropout', .20,.75),
            'recurrent_dropout': hpo.hp.uniform('recurrent_dropout', .20, 75),
            'optimizer': hpo.hp.choice('optimizer', ['rmsprop', 'adadelta', 'adam']),
            'epochs' : hpo.hp.choice('epochs', 15, 30),
            'batch_size' : hpo.hp.choice('batch_size', [32, 64, 128, 512])
    }

    model.fit(x_train, y_train,
            batch_size=params['batch_size'],
            epochs=15,
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=params['batch_size'])
    
    return {'loss': -acc, 'status': hpo.STATUS_OK}

trials=hpo.Trials()
best = nu_fmin("hello", objective, space, algo=hpo.tpe.suggest, max_evals=50, trials=trials)