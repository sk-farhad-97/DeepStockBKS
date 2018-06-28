import subprocess
import sys
import os
import signal
from websocket import create_connection
import requests
from config import python_comm

PORT = 8888
HOST = 'localhost'

if len(sys.argv) < 12:
    ws = create_connection("ws://" + str(HOST) + ":" + str(PORT) + "/ws")
    ws.send("$> Incomplete arguments! exiting.........")
    exit(1)


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
file_name = sys.argv[11]


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True


def start_process():
    killer = GracefulKiller()
    ws = create_connection("ws://" + str(HOST) + ":" + str(PORT) + "/ws")
    process = subprocess.Popen(
        [
            python_comm,
            file_name,
            MODEL_NAME,
            DATA_FILE,
            REWARD_FUNC,
            SYMBOL,
            TRAIN_INI,
            TRAIN_FI,
            TEST_INI,
            TEST_FI,
            EPOCH,
            FEATURE_LIST
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    pid = process.pid
    ws.send("$> Training Started with id: " + str(pid))
    line = process.stdout.readline()
    while line:
        if killer.kill_now:
            ws.send('$>Killed!!!')
            os.kill(pid, signal.SIGKILL)
        line = process.stdout.readline()
        ws.send("$> " + line.rstrip().decode("utf-8") + "\n")

    err = process.stderr.read()
    if err:
        ws.send("$> System output==>")
        ws.send("$> " + err.rstrip().decode("utf-8") + "\n")
        ws.send("$> Training finished!!!")
    else:
        ws.send("$> Training finished!")
    r = requests.post('http://' + HOST + ':' + str(PORT) + '/stop_training', data={'key': 'value'})


if __name__ == "__main__":
    start_process()



