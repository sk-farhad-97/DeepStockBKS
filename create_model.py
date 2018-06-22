import subprocess
import sys
import os
import signal
from websocket import create_connection
import requests

PORT = 8888
HOST = 'localhost'

if len(sys.argv) < 5:
    ws = create_connection("ws://" + str(HOST) + ":" + str(PORT) + "/ws")
    ws.send("$> Incomplete arguments! exiting.........")
    exit(1)


MODEL_NAME = sys.argv[1]
NUM_FEATURES = sys.argv[2]
DROPOUT = sys.argv[3]
file_name = sys.argv[4]


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
            'python3',
            file_name,
            MODEL_NAME,
            NUM_FEATURES,
            DROPOUT
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    pid = process.pid
    ws.send("$> Model creation Started with id: " + str(pid))
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
        ws.send("$> Model creation finished!!!")
    else:
        ws.send("$> Model creation finished!")
    r = requests.post('http://'+HOST + ':' + str(PORT) + '/stop_training', data={'key': 'value'})


if __name__ == "__main__":
    start_process()



