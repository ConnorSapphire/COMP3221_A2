import sys
import socket
import threading
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
        self.E = 10

    def start(self):
        print(f"I am client {self.client_id.strip("client")}")
        self.retrieve_data()
        self.listener_thread = threading.Thread(target=self.listen_to_server)
        self.listener_thread.start()
        self.send(f"CONNECTION ESTABLISHED")

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
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.bind((HOST, self.port))
                client_socket.listen()
                print(f"Client listening on port {self.port}")

                conn, addr = client_socket.accept()
                with conn:
                    data = conn.recv(1048)
                    if data:
                        print("Client received data")
                try:
                    d = pickle.loads(data)
                    print(d["w"])
                    print(d["b"])
                    print("Received data")
                except Exception as e:
                    print(f"Failed: {e}")
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Can't connect to the listener socket: {e}")

    def send(self, message):
        message = {
            "client_id": self.client_id,
            "port": self.port,
            "content": message,
        }

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.connect((HOST, SERVER_PORT))
                server_socket.sendall(json.dumps(message).encode())
                print("Message sent")
        except KeyboardInterrupt:
            exit()
        except Exception:
            print("Message failed")
    
    def evaluate(self):
        pass
    
    def update(self):
        if self.opt_method == 0:
            self.gradient_descent()
        else:
            self.mini_batch()
    
    def gradient_descent(self):
        pass
    
    def mini_batch(self):
        pass

    def retrieve_data(self):
        # retrieve training data
        df = pd.read_csv(f"./FLData/calhousing_train_{self.client_id}.csv")
        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values
        self.X_train = torch.Tensor(X_train).type(torch.float32)
        self.Y_train = torch.Tensor(y_train).type(torch.float32)
        
        
        # x = [i[0] for i in self.X_train]
        # plt.scatter(x, self.Y_train, label="Initial Data")
        # plt.title("Pre Pytorch")
        # plt.xlabel("X")
        # plt.ylabel("y")
        # plt.legend()
        # plt.show()
        
        # retrieve testing data
        df = pd.read_csv(f"./FLData/calhousing_test_{self.client_id}.csv")
        X_test = df.iloc[:, :-1].values
        y_test = df.iloc[:, -1].values
        self.X_test = torch.Tensor(X_test).type(torch.float32)
        self.Y_test = torch.Tensor(y_test).type(torch.float32)
                
    
if __name__ == "__main__":
    id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    client = Client(id, port, opt_method)
    client.start()