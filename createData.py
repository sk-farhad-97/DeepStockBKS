from StockLSTMmethods_3C import *


np.random.seed(1335)  # for reproducibility
dir_path = os.path.dirname(os.path.realpath(__file__))


def create_price():
    n_up = 150
    n_down = 150
    sine_u = 6
    sine_d = 6.5
    price_sigma = 0.2
    price_base = 50

    price_up1 = price_sigma * (-np.sin(np.arange(n_up) / sine_u) +
               (np.arange(n_up) / 20.0)) + price_sigma * 0.1 * np.random.randint(0, 10, size=n_up)  # prices with up-trend

    price_down1 = price_up1[::-1] + price_sigma * (np.sin(np.arange(n_down) / sine_d) -
                 (np.arange(n_down) / 20.0)) + price_sigma * 0.1 * np.random.randint(0, 10, size=n_up)  # prices with down-trend

    price_up2 = price_down1[::-1] + price_sigma * (np.sin(np.arange(n_up) / sine_d) +
               (np.arange(n_up) / 20.0)) + price_sigma * 0.1 * np.random.randint(0, 10, size=n_up)

    price = np.append(np.append(price_up1, price_down1), price_up2) + price_base

    data = np.array([['', 'Close', 'diff']])
    for i in range(0, len(price)):
        diff = 0
        if i>1:
            diff = price[i] - price[i-1]
        data = np.insert(data, i + 1, np.array((i, price[i-1], diff)), 0)

    df = pd.DataFrame(data=data[1:, 1:],
                  index=data[1:, 0],
                  columns=data[0, 1:])
    print(df)
    return price, df


def get_csv_data():
    df = pd.read_csv(os.path.join(dir_path, 'data', 'RL_data.csv'))
    df_acv = df.loc[df['Symbol'] == 'ACV']
    #print(df_acv)
    return df_acv


def load_data(test=False):
    df = pd.read_csv(os.path.join(dir_path, 'data', 'RL_data.csv'))
    data = df.loc[df['Symbol'] == 'ACV']

    x_train = data.iloc[:400,]
    x_test= data.iloc[350:450,]
    if test:
        return x_test
    else:
        return x_train


if __name__ == '__main__':

    indata = load_data()
    prices = indata['Close'].values.astype(float)
    # prices, df = create_price()
    state, trade_info, xdata_trf, close = init_state(indata)
    # xdata_trf[2:3, 0:1, :][0][0][0] = -0.5
    # state = xdata_trf[2:3, 0:1, :]

    print(state)
    # plt.plot(prices)
    # plt.ylabel(' Price')
    # plt.show()