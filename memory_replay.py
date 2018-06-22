import random
import numpy as np

batchSize = 25
buffer = 50
gamma = 0.6  # since the reward can be several time steps away, make gamma high


def exp_replay(h, replay, state, action, reward, new_state, MODEL, CHOICES, terminal_state):
    # Experience replay storage
    if len(replay) < buffer:  # if buffer not filled, add to it
        replay.append((state, action, reward, new_state))
    else:  # if buffer full, overwrite old values
        if h < (buffer - 1):
            h += 1
        else:
            h = 0
        replay[h] = (state, action, reward, new_state)
        # randomly sample our experience replay memory
        minibatch = random.sample(replay, batchSize)
        X_train = []
        y_train = []
        for memory in minibatch:
            # Get max_Q(S',a)
            old_state, action, reward, new_state = memory
            old_qval = MODEL.predict(old_state, batch_size=1)
            newQ = MODEL.predict(new_state, batch_size=1)
            maxQ = np.max(newQ)
            # number of choices 3
            y = np.zeros((1, CHOICES))
            y[:] = old_qval[:]
            if terminal_state == 0:  # non-terminal state
                update = (reward + (gamma * maxQ))
            else:  # terminal state
                update = reward
            y[0][action] = update
            # print(time_step, reward, terminal_state)
            X_train.append(old_state)
            y_train.append(y.reshape(CHOICES, ))

        X_train = np.squeeze(np.array(X_train), axis=(1))
        y_train = np.array(y_train)
        MODEL.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)

    return h, replay, MODEL