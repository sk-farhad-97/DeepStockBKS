import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
from model_io import save_model

CHOICES = 3
NUM_FEATURES = 6 # number of rows in init_state() xdata
# DROPOUT = 0.5
model_path = 'models/'

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
    NUM_FEATURES = int(sys.argv[2])
    DROPOUT = float(sys.argv[3])


def create_model():
    model = Sequential()
    model.add(LSTM(64,
                   input_shape=(1, NUM_FEATURES),
                   return_sequences=True,
                   stateful=False))
    model.add(Dropout(DROPOUT))

    model.add(LSTM(64,
                   input_shape=(1, NUM_FEATURES),
                   return_sequences=False,
                   stateful=False))
    model.add(Dropout(DROPOUT))

    model.add(Dense(CHOICES, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    rms = RMSprop()
    adam = Adam()
    model.compile(loss='mse', optimizer=adam)
    return model


print('Creating model')
model = create_model()
saved = save_model(model, model_path, MODEL_NAME)
if saved:
    print('Model Created successfully!!')
else:
    print('Model was not created!!')
