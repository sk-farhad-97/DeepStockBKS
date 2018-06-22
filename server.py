import os
import json
import signal
from tornado.websocket import WebSocketHandler
from tornado.web import RequestHandler, Application
from tornado import httpserver, ioloop
import socket
import subprocess

clients = dict()
port = 8888
host = 'localhost'

'''rabbit MQ codes'''
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CURRENT_PROCESS = -1

message_global = ""

'''web socket codes'''
active_clients = set()


class WSHandler(WebSocketHandler):
    def open(self, *args):
        print("New connection")
        active_clients.add(self)
        # print clients

    def on_message(self, message):
        for client in active_clients:
            if client != self:
                client.write_message(message)

    def on_close(self):
        print('connection closed')
        active_clients.remove(self)

    def check_origin(self, origin):
        return True


'''Web client'''


class HomeHandler(RequestHandler):
    def get(self):
        self.render("templates/homepage.html")


class StartTrainingHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")

    def post(self):
        global CURRENT_PROCESS
        if CURRENT_PROCESS != -1:
            print("Another Process running. PID: ", CURRENT_PROCESS)
        else:
            req_body = json.loads(self.request.body.decode())
            print(req_body)
            process = subprocess.Popen(
                [
                    'python3',
                    "start_training.py",
                    req_body['model'],
                    req_body['data'],
                    req_body['reward'],
                    req_body['symbol'],
                    req_body['train_ini'],
                    req_body['train_fi'],
                    req_body['test_ini'],
                    req_body['test_fi'],
                    req_body['epoch'],
                    req_body['feature_list'],
                    'trainLSTM_3C.py',
                ],
                stdout=subprocess.PIPE
            )
            CURRENT_PROCESS = process.pid
            self.write("Process started")


class StartEvaluationHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")

    def post(self):
        global CURRENT_PROCESS
        req_body = json.loads(self.request.body.decode())
        if CURRENT_PROCESS != -1:
            print("Another Process running. PID: ", CURRENT_PROCESS)
        else:
            print(req_body)
            process = subprocess.Popen(
                [
                    'python3',
                    "start_evaluation.py",
                    req_body['model'],
                    req_body['data'],
                    'unrealized_pnl',
                    req_body['symbol'],
                    req_body['test_ini'],
                    req_body['test_fi'],
                    "evaluate_models.py",
                ],
                stdout=subprocess.PIPE
            )
            CURRENT_PROCESS = process.pid
            self.write("Evaluation started")


class ListModelsHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")

    def get(self):
        models = []
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".json"):
                models.append(file)
        models = {
            'model_list': models
        }
        self.write(json.dumps(models))


class ListDataFilesHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")

    def get(self):
        files = []
        for file in os.listdir(DATA_DIR):
            if file.endswith(".csv"):
                files.append(file)
        files = {
            'file_list': files
        }
        self.write(json.dumps(files))


class StopTrainingHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")

    def post(self):
        global CURRENT_PROCESS
        print('Open process: ', CURRENT_PROCESS)
        if CURRENT_PROCESS != -1:
            os.kill(CURRENT_PROCESS, signal.SIGTERM)
            print('Stopped: ', CURRENT_PROCESS)
            CURRENT_PROCESS = -1

        self.write('STOP')


application = Application([
    (r'/ws', WSHandler),
    (r"/run_training", StartTrainingHandler),
    (r"/stop_training", StopTrainingHandler),
    (r"/run_evaluation", StartEvaluationHandler),
    (r"/stop_evaluation", StopTrainingHandler),
    (r"/home", HomeHandler),
    (r"/models", ListModelsHandler),
    (r"/datafiles", ListDataFilesHandler),
])

if __name__ == "__main__":
    http_server = httpserver.HTTPServer(application)
    http_server.listen(port)
    myIP = socket.gethostbyname(socket.gethostname())
    print('*** Websocket Server Started at: ***', myIP)

    ioloop.IOLoop.instance().start()
