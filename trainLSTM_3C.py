import sys
import random, timeit
from keras.optimizers import Adam
from model_factory import create_model, CHOICES
from memory_replay import exp_replay
from model_io import save_model, load_model
from StockLSTMmethods_3C import *


if len(sys.argv) < 11:
    print('Usage: python3 trainLSTM_3C.py $model_name $datafile_name $reward_func $symbol $train_ini $train_fi $test_ini $test_fi')
    print('Available reward functions: valid_seq, strategy')
    exit(1)


epochs = 100
epsilon = 1
start_time = timeit.default_timer()

MODEL_NAME = sys.argv[1]
DATA_FILE = sys.argv[2]
REWARD_FUNC = sys.argv[3]
SYMBOL = sys.argv[4]
TRAIN_INI = sys.argv[5]
TRAIN_FI = sys.argv[6]
TEST_INI = sys.argv[7]
TEST_FI = sys.argv[8]
EPOCH = sys.argv[9]
FEATURE_LIST = sys.argv[10]

print('STD out....')

if int(EPOCH) > 0:
    epochs = int(EPOCH)

if len(FEATURE_LIST) < 1:
    FEATURE_LIST = ['m3', 'm3_to_m1']
    print('Empty feature list!')
    exit(1)
else:
    print('Feature list: ' + FEATURE_LIST)
    FEATURE_LIST = FEATURE_LIST.split(',')

model_path = 'models/'

MODEL = load_model(model_path, MODEL_NAME)
if not MODEL:
    print('Model not found! Exiting.........')
    exit(1)
    
adam = Adam()
MODEL.compile(loss='mse', optimizer=adam)



'''Load data'''
DATA_FILE += '.csv'
indata = load_data_v2(DATA_FILE, SYMBOL, [TRAIN_INI, TRAIN_FI], [TEST_INI, TEST_FI])
test_data = load_data_v2(DATA_FILE, SYMBOL, [TRAIN_INI, TRAIN_FI], [TEST_INI, TEST_FI], test=True)
replay = []
learning_progress = []
h = 0
signal = np.zeros(len(indata))


for i in range(epochs):
    if i == 0:
        pass

    state, trade_info, xdata_trf, price_data = init_state_v2(indata, FEATURE_LIST)
    terminal_state = 0
    time_step = 1
    print('Training Model............')
    while terminal_state == 0:
        state = xdata_trf[time_step - 1:time_step, 0:1, :]
        qval = MODEL.predict(state, batch_size=1)
        if (random.random() < epsilon) and i != epochs - 1:
            action = np.random.randint(0, CHOICES)  # assumes different actions [0, CHOICES)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))
        # Take action, observe new state S'
        new_trade_info, new_xdata_trf, new_action, signal, terminal_state = \
            take_action(trade_info, xdata_trf, action, signal, time_step)
        # Observe reward
        reward = get_reward(REWARD_FUNC, new_trade_info, time_step, new_action, signal, price_data, terminal_state)
        # Get max_Q(S',a)

        new_state = new_xdata_trf[time_step:time_step + 1, 0:1, :]

        # Experience replay storage
        h, replay, MODEL = exp_replay(h, replay, state, action, reward,
                   new_state, MODEL, CHOICES, terminal_state)
        if new_action != -1:
            trade_info = new_trade_info
            xdata_trf = new_xdata_trf
            time_step += 1
    print('Evaluating............')
    eval_reward, cash_gained, predictions = evaluate_Q(test_data, MODEL, i)
    # print(predictions)
    learning_progress.append((eval_reward))
    print("Epoch #: %s Reward: %f Cash: %f" % (i, eval_reward, cash_gained))
    # learning_progress.append((reward))
    save_model(MODEL, model_path, MODEL_NAME)
    if epsilon > 0.1:  # decrement epsilon over time
        epsilon -= (1.0 / epochs)

elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

