SCALE_FACTOR = 10


def reward_function(action, trade_info, time_step, stock_slot):
    # print('valid swq...')
    buy_pre = trade_info[time_step - 1][1]
    if action == 0:
        reward = .1
    elif action == 1:
        reward = .1
    elif action == 2:
        if buy_pre > 0:
            reward = 0
        else:
            reward = 0
    else:
        '''Forbidden action'''
        reward = -.05

    return reward


def reward_function_urpnl(action, trade_info, time_step, stock_slot):
    # print('strategy.....')
    close_cur = trade_info[time_step][0]
    close_pre = trade_info[time_step - 1][0]
    buy_cur = trade_info[time_step][1]
    buy_pre = trade_info[time_step - 1][1]
    cash_cur = trade_info[time_step][2]
    cash_pre = trade_info[time_step - 1][2]

    if action == 0:
        reward = SCALE_FACTOR*(((close_cur - close_pre) * stock_slot) / cash_pre) # unrealized PNL
    elif action == 1:
        reward = SCALE_FACTOR*(((close_cur - buy_pre) * stock_slot) / cash_pre) # realized PNL
    elif action == 2:
        if buy_pre>0:
            reward = SCALE_FACTOR*((-buy_pre * 0.25 * stock_slot) / cash_pre)
        else:
            reward = 0
    else:
        '''Forbidden action'''
        reward = -.05

    return reward

