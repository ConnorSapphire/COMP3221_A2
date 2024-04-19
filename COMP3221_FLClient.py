import sys
import socket
import threading
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

HOST = "localhost"
SERVER_PORT = 6000

class Client:
    def __init__(self, id, port, opt_method):
        self.client_id = id
        self.port = port
        self.opt_method = opt_method
        self.stop_event = threading.Event()
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None
        self.opt = None
        self.loss_fn = F.mse_loss
        self.epochs = 100
        self.learning_rate = 1e-10

    def start(self):
        print(f"I am client {self.client_id.strip("client")}")
        self.retrieve_data()
        self.listener_thread = threading.Thread(target=self.listen_to_server)
        self.listener_thread.start()
        self.send_message(f"CONNECTION ESTABLISHED")

    def stop(self):
        try:
            self.stop_event.set()
            self.listener_thread.join()
        except KeyboardInterrupt:
            exit()
        except Exception:
            pass

    def listen_to_server(self):  # Listen on port 6001, 6002, etc.
        try:
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.bind((HOST, self.port))
                    client_socket.listen()
                    print(f"Client listening on port {self.port}")

                    conn, addr = client_socket.accept()
                    data = b""
                    with conn:
                        while True:
                            packet = conn.recv(1048)
                            if not packet:
                                break
                            data += packet
                        if data:
                            print("Client received data")
                    try:
                        model = pickle.loads(data)
                        self.model = model["model"]
                        self.opt = optim.SGD(self.model.parameters(), lr=self.learning_rate)
                        print("Received data")
                        self.update()
                    except Exception as e:
                        print(f"Failed: {e}")
                    client_socket.close()
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Can't connect to the listener socket: {e}")

    def send_message(self, message):
        message = {
            "client_id": self.client_id,
            "port": self.port,
            "data_size": list(self.X_train.size())[0],
            "content": message,
        }

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((HOST, SERVER_PORT))
                client_socket.sendall(b"0")
                client_socket.sendall(json.dumps(message).encode())
                print("Message sent")
                client_socket.close()
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Message failed: {e}")
            
    def send_model(self):
        message = {
            "client_id": self.client_id,
            "model": self.model,
        }

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((HOST, SERVER_PORT))
                client_socket.sendall(b"1")
                client_socket.sendall(pickle.dumps(message))
                print("Model sent")
                client_socket.close()
        except KeyboardInterrupt:
            exit()
        except Exception:
            print("Model failed to send")
    
    def evaluate(self):
        pred = self.model(self.X_test)
        loss = self.loss_fn(pred, self.Y_test)
        return loss
    
    def update(self):
        if self.opt_method == 0:
            self.gradient_descent()
        else:
            self.mini_batch()
        self.send_model()
    
    def gradient_descent(self):
        for e in range(self.epochs):
            pred = self.model(self.X_train)
            loss = self.loss_fn(pred, self.Y_train)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
    
    def mini_batch(self):
        pass

    def retrieve_data(self):
        # retrieve training data
        df = pd.read_csv(f"./FLData/calhousing_train_{self.client_id}.csv")
        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values
        self.X_train = torch.Tensor(X_train).type(torch.float32)
        self.Y_train = torch.Tensor(y_train).type(torch.float32).unsqueeze(1)
        
        # retrieve testing data
        df = pd.read_csv(f"./FLData/calhousing_test_{self.client_id}.csv")
        X_test = df.iloc[:, :-1].values
        y_test = df.iloc[:, -1].values
        self.X_test = torch.Tensor(X_test).type(torch.float32)
        self.Y_test = torch.Tensor(y_test).type(torch.float32).unsqueeze(1)
                
    
if __name__ == "__main__":
    id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    client = Client(id, port, opt_method)
    client.start()