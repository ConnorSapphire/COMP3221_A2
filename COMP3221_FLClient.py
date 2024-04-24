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
import time

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
        self.epochs = (opt_method == 0 and 300) or 100
        self.learning_rate = 1e-7
        self.confirmed = threading.Event()
        self.iteration = 0
        
    def start(self) -> None:
        """
        Start all processes for client instance. Including retreiving data, creating log
        files, creating the listening socket thread, and sending the handshake message to the
        server.
        """
        print(f"I am client {self.client_id.strip('client')}")
        self.create_log()
        self.retrieve_data()
        self.write_log(f"I am client {self.client_id.strip('client')}")
        self.write_log(f"Learning rate: {self.learning_rate}")
        self.write_log(f"Epochs: {self.epochs}")
        self.write_log(f"Optimisation method: {'mini-batch ' if self.opt_method == 1 else ''}gradient descent")
        self.listener_thread = threading.Thread(target=self.listen_to_server)
        self.listener_thread.start()
        self.send_message(f"CONNECTION ESTABLISHED")

    def stop(self) -> None:
        """
        Stop the listener thread.
        """
        try:
            self.stop_event.set()
            self.listener_thread.join()
        except Exception:
            pass

    def listen_to_server(self) -> None:  # Listen on port 6001, 6002, etc.
        """
        Listen to the server for incoming messages across the socket using the 
        client's port.
        """
        print(f"Client listening on port {self.port}")
        while not self.stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.bind((HOST, self.port))
                    client_socket.listen(5)
                    conn, addr = client_socket.accept()
                    with conn:
                        try:
                            # first bit represents type of message
                            data = conn.recv(1)
                            # check if expecting a model
                            if data == b"0": # expecting model
                                data = b""
                                while True:
                                    packet = conn.recv(1048)
                                    if not packet:
                                        break
                                    data += packet
                                if data:
                                    try:
                                        model = pickle.loads(data)
                                        self.model = model["model"]
                                        self.opt = optim.SGD(self.model.parameters(), lr=self.learning_rate)
                                        self.iteration = model["iteration"]
                                        # self.confirmed = False
                                        print(f"\nI am client {self.client_id.strip('client')}")
                                        print(f"Received new global model {self.iteration + 1}")
                                        self.write_log(f"\nReceived new global model {self.iteration + 1}")
                                        update = threading.Thread(target=self.update)
                                        update.start()
                                    except Exception as e:
                                        print(f"Failed: {e}")
                            # check if confirming a model was received by the server
                            elif data == b"1": # confirming model received
                                self.confirmed.set()
                        except Exception as e:
                            print("Could not read from server: {e}")
                    client_socket.close()
            except Exception as e:
                print(f"Can't connect to the listener socket: {e}")

    def send_message(self, message: str) -> None:
        """
        Send a message to the server. Used for sending the handshake method to tell the server
        the client is available.

        Args:
            message (str): String message to be sent to the server.
        """
        # define message
        message = {
            "client_id": self.client_id,
            "port": self.port,
            "data_size": list(self.X_train.size())[0],
            "content": message,
        }

        # attempt to send message
        self.confirmed.clear()
        sent = False
        while not sent:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((HOST, SERVER_PORT))
                    # send binary 0 first to inform server to expect message in this format
                    client_socket.sendall(b"0") 
                    # send message
                    client_socket.sendall(json.dumps(message).encode())
                    # wait and then confirm whether message was received
                    if self.confirmed.wait(timeout=0.25):
                        sent = True
                        print(f"Message sent to server: {message['content']}")
                        self.confirmed.clear()
                    client_socket.close()
            except Exception as e:
                print(f"Message failed: {e}")
            
    def send_model(self) -> None:
        """
        Send the model back to the server. Should be called after the model has been trained.
        """
        # define the message
        message = {
            "client_id": self.client_id,
            "iteration": self.iteration,
            "model": self.model,
        }
        
        # attempt to send the message
        self.confirmed.clear()
        sent = False
        while not sent:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((HOST, SERVER_PORT))
                    # send binary 1 first to inform server to expect message in this format
                    client_socket.sendall(b"1")
                    # send the message
                    client_socket.sendall(pickle.dumps(message))
                    # wait and then confirm whether message was received
                    if self.confirmed.wait(timeout=0.25):
                        sent = True
                    client_socket.close()
            except Exception as e:
                print(f"Model failed to send: {e}")
        print("Sending new local model")
    
    def evaluate(self) -> torch.Tensor:
        """
        Test the model against the testing data using Mean Squared Error (MSE).

        Returns:
            torch.Tensor: The MSE error of the current model on the testing data.
        """
        pred = self.model(self.X_test)
        loss = self.loss_fn(pred, self.Y_test)
        print(f"Testing MSE: {loss:.04f}")
        self.write_log(f"\tTesting MSE: {loss:.04f}")
        return loss
    
    def update(self) -> None:
        """
        Updates the global model using the client instances specific update method.
        Either gradient descent or mini-batch gradient descent. Sends the model back
        to the server once the training is complete.
        """
        self.evaluate()
        print(f"Local training...")
        if self.opt_method == 0:
            self.gradient_descent()
        else:
            self.mini_batch()
        self.send_model()
    
    def gradient_descent(self) -> None:
        """
        Performs the gradient descent algorithm on the current model.
        This occurs multiple times, as determined by self.epochs.
        The MSE result before and after all epochs are printed to the terminal and
        saved in the logs.
        """
        losses = []
        #start_time = time.time()
        for e in range(self.epochs):
            start_time = time.time()
            pred = self.model(self.X_train)
            loss = self.loss_fn(pred, self.Y_train)
            losses.append(loss)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        #epoch_duration = time.time() - start_time
        #print(f"{self.epochs} epochs completed in {epoch_duration:.2f} seconds")

        print(f"Training MSE: {losses[-1]:.04f}")
        self.write_log(f"\tPre-update training MSE: {losses[0]:.04f}")
        self.write_log(f"\tPost-update training MSE: {losses[-1]:.04f}")
    
    def mini_batch(self):
        """
        Performs the mini-batch gradient descent algorithm on the current model.
        This occurs multiple times, as determined by self.epochs.
        The MSE result before and after all epochs are printed to the terminal and
        saved in the logs.
        """
        batch_size = 64
        dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        #start_time = time.time()
        for e in range(self.epochs):
            epoch_losses = []
            for X_batch, y_batch in dataloader:
                pred = self.model(X_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                epoch_losses.append(loss.item())
            losses.append(sum(epoch_losses) / len(epoch_losses))

        #epoch_duration = time.time() - start_time
        #print(f"{self.epochs} epochs completed in {epoch_duration:.2f} seconds")

        print(f"Training MSE: {losses[-1]:.04f}")
        self.write_log(f"\tPre-update training MSE: {losses[0]:.04f}")
        self.write_log(f"\tPost-update training MSE: {losses[-1]:.04f}")

    def retrieve_data(self):
        """
        Retrieves the training data and testing data for the client from the files.
        This data is converted into Tensors and stored inside the client instance then
        used for evaluating and training the model in each iteration.
        """
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
        print("Data retrieved from files")
        
    def create_log(self) -> None:
        """
        Creates and clears the log file for this specific instance of the client. The log file
        is stored at:
        ./FLLogs/<client_id>_log.txt
        for each client.
        """
        try:
            with open(f"./FLLogs/{self.client_id}_log.txt", "w") as f:
                f.close()
        except IOError as e:
            print(f"Error creating log file: {e}")
        
    def write_log(self, message: str) -> None:
        """
        Appends a message to the log file for this specific instance of the client, includes a
        new line. The log file is stored at:
        ./FLLogs/<client_id>_log.txt
        for each client.
        """
        try:
            with open(f"./FLLogs/{self.client_id}_log.txt", "a") as f:
                f.write(message + "\n")
                f.close()
        except IOError as e:
            print(f"Error with writing to log: {e}")
    
if __name__ == "__main__":
    id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    client = Client(id, port, opt_method)
    client.start()