from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

CHOICES = 3
num_features = 6 # number of rows in init_state() xdata


def create_model():
    model = Sequential()
    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=True,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=False,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(Dense(CHOICES, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    rms = RMSprop()
    adam = Adam()
    model.compile(loss='mse', optimizer=adam)
    return model
