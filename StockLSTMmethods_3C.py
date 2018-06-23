from __future__ import print_function
import numpy as np
import os
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.externals import joblib
from reward_factory import reward_function, reward_function_urpnl
from config import STOCK_SLOT, INIT_CASH
# import quandl
# import talib

np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

dir_path = os.path.dirname(os.path.realpath(__file__))


# Load data
def load_data_v2(file_name, stock_sym, train_range, test_range, test=False):
    df = pd.read_csv(os.path.join(dir_path, 'data', file_name))
    data = df.loc[df['Symbol'] == stock_sym]
    # print(train_range, test_range)
    x_train = data.iloc[int(train_range[0]):int(train_range[1]),]
    x_test= data.iloc[int(test_range[0]):int(test_range[1]),]
    if test:
        return x_test
    else:
        return x_train


def load_data_eval(file_name, stock_sym, test_range):
    df = pd.read_csv(os.path.join(dir_path, 'data', file_name))
    data = df.loc[df['Symbol'] == stock_sym]
    # print(train_range, test_range)
    x_test = data.iloc[int(test_range[0]):int(test_range[1]),]
    return x_test


def get_state(xdata, t_step):
    scaler = joblib.load('data/scaler.pkl')
    xdata_trf = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    state = xdata_trf[t_step:t_step + 1, 0:1, :]
    return state


# Initialize first state, all items are placed deterministically
def init_state_v2(indata, columns, test=False):
    close = indata['Close'].values.astype(float) # convert to float for ndarray
    buy_price = np.zeros(len(close))
    cash = np.empty(len(close))
    cash.fill(INIT_CASH)
    status = np.empty(len(close))
    status.fill(0)

    col_stack_list = [status]

    for col in columns:
        col_stack_list.append(indata[col].values.astype(float))

    #--- Preprocess data
    trade_info = np.column_stack((close, buy_price, cash))
    xdata = np.column_stack((col_stack_list))
    trade_info = np.nan_to_num(trade_info)
    xdata = np.nan_to_num(xdata)
    xdata_trf = None
    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata_trf = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    elif test == True:
        scaler = joblib.load('data/scaler.pkl')
        xdata_trf = np.expand_dims(scaler.fit_transform(xdata), axis=1)

    state = xdata_trf[0:1, 0:1, :]

    return state, trade_info, xdata_trf, close


# Take Action
def take_action(trade_info, xdata_trf, action, signal, time_step, eval_data=False):

    if time_step + 1 >= trade_info.shape[0]:
        terminal_state = 1
        signal[time_step] = 0
        return trade_info, xdata_trf, 2, signal, terminal_state

    close_cur = trade_info[time_step][0]
    close_pre = trade_info[time_step - 1][0]
    buy_pre = trade_info[time_step - 1][1]
    cash_pre = trade_info[time_step - 1][2]

    if action == 0:
        # sell
        xdata_trf[time_step:time_step + 1, 0:1, :][0][0][0] = -0.5
        if buy_pre > 0:
            signal[time_step] = -STOCK_SLOT
            trade_info[time_step][1] = 0
            trade_info[time_step][2] = cash_pre + (STOCK_SLOT * close_cur)
        else:
            if not eval_data:
                action = -1
            else:
                signal[time_step] = -STOCK_SLOT
                trade_info[time_step][1] = 0
                trade_info[time_step][2] = cash_pre + (STOCK_SLOT * close_cur)

    elif action == 1:
        # buy
        xdata_trf[time_step:time_step + 1, 0:1, :][0][0][0] = 0.5
        if buy_pre == 0:
            signal[time_step] = STOCK_SLOT
            trade_info[time_step][1] = close_cur
            trade_info[time_step][2] = cash_pre - (STOCK_SLOT * close_cur)
        else:
            if not eval_data:
                action = -1
            else:
                signal[time_step] = STOCK_SLOT
                trade_info[time_step][1] = close_cur
                trade_info[time_step][2] = cash_pre - (STOCK_SLOT * close_cur)
    else:
        # none/keep
        signal[time_step] = 0
        trade_info[time_step][1] = buy_pre
        trade_info[time_step][2] = cash_pre

    terminal_state = 0
    return trade_info, xdata_trf, action, signal, terminal_state


# Get Reward, the reward is returned at the end of an episode
def get_reward(REWARD_FUNC, trade_info, time_step, action, price_data, signal, terminal_state, eval=False, epoch=0):
    if REWARD_FUNC == 'valid_sequence':
        reward = reward_function(action, trade_info, time_step, STOCK_SLOT)
    if REWARD_FUNC == 'unrealized_pnl':
        reward = reward_function_urpnl(action, trade_info, time_step, STOCK_SLOT)

    if terminal_state == 1 and eval:
        print('Saving plot.....')
        # save a figure of the test set
        signal = pd.Series(data=[x for x in signal], index=np.arange(len(signal)))
        bt = twp.Backtest(pd.Series(data=[x for x in price_data], index=signal.index.values),
                          signal, signalType='shares')

        plt.figure(figsize=(3, 4))
        bt.plotTrades(trade_info[time_step - 1][2])
        plt.axvline(x=400, color='black', linestyle='--')
        plt.suptitle(str(epoch))
        plt.savefig('plt/' + str(epoch) + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
        plt.close('all')

    return reward


def evaluate_Q(FEATURE_LIST, eval_data, eval_model, epoch=0, eval=False):
    signal = np.zeros(len(eval_data))
    state, trade_info, xdata_trf, price_data = init_state_v2(eval_data, FEATURE_LIST)
    terminal_state = 0
    time_step = 1
    eval_reward = 0
    choices = ['buy', 'sell', 'keep/none']
    predictions = []
    while terminal_state == 0:
        qval = eval_model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        # Take action, observe new state S'
        predictions.append(choices[action])
        trade_info, xdata_trf, action, signal, terminal_state = \
            take_action(trade_info, xdata_trf, action, signal, time_step,  eval_data=True)
        # Observe reward
        # trade_info, time_step, action, price_data, signal, terminal_state,
        eval_reward += get_reward('unrealized_pnl', trade_info, time_step, action, price_data, signal, terminal_state, eval=eval,
                                 epoch=epoch)
        state = xdata_trf[time_step:time_step + 1, 0:1, :]
        if terminal_state == 0:  # terminal state
            time_step += 1
        else:
            qval = eval_model.predict(state, batch_size=1)
            action = (np.argmax(qval))
            # Take action, observe new state S'
            predictions.append(choices[action])

    return eval_reward, trade_info[time_step - 1][2], predictions