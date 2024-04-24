import sys
import socket
import threading
import json
import torch
import torch.nn as nn
import pickle
import random
import time

HOST = "localhost"

class Server:
    def __init__(self, port, subsamp):
        self.port = port
        self.subsamp = subsamp
        self.clients = {}
        self.subsamp_clients = []
        self.client_stack = {}
        self.listener_threads = []
        self.federated_thread = None
        self.stop_event = threading.Event()
        self.iteration = 0
        self.T = 100
        self.model = nn.Linear(8, 1)
        self.wait = 30

    def start(self):
        """
        Start all processes for server instance. Including creating the algorithm thread, and 
        creating the listening socket thread.
        """
        self.federated_thread = threading.Thread(target=self.federate, daemon=True)
        listener_thread = threading.Thread(target=self.listen_to_client)
        self.listener_threads.append(listener_thread)
        listener_thread.start()

    def stop(self):
        """
        Stop the listener threads.
        """
        try:
            self.stop_event.set()
            for thread in self.listener_threads:
                thread.join()
        except:
            pass

    def listen_to_client(self):  # Listen on port 6000
        """
        Listen for messages from the clients and interpret them. Messages will be sent to
        the server instances port.
        """
        print(f"Server listening on port {self.port}")
        while not self.stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.bind((HOST, self.port))
                    server_socket.listen(5)
                    conn, addr = server_socket.accept()
                    with conn:
                        try:
                            message = conn.recv(1)
                            # check if message is a string
                            if message == b"0": # message is a string
                                message = conn.recv(1024)
                                while True:
                                    packet = conn.recv(1024)
                                    if not packet:
                                        break
                                    message += packet
                                message = json.loads(message.decode("utf-8"))
                                client_id = message["client_id"]
                                client_data_size = message["data_size"]
                                client_port = message["port"]
                                content = message["content"]
                                if message['content'] == "CONNECTION ESTABLISHED":
                                    self.send_confirmation(client_port)
                                    if len(self.client_stack) < 5:
                                        print(f"== Handshake: handle {client_id} connection ==")
                                        self.client_stack[client_id] = {"port": client_port, "data_size": client_data_size}
                                        if not self.federated_thread.is_alive():
                                            self.federated_thread.start()
                                    else:
                                        print("== Client ignored : Too many clients ==")
                                else:
                                    print(f"\tMessage from {client_id} on port {client_port}: {content}")
                            # check if message is a model
                            elif message == b"1": # message is a model
                                data = b""
                                while True:
                                    packet = conn.recv(1048)
                                    if not packet:
                                        break
                                    data += packet
                                if data:
                                    try:
                                        model = pickle.loads(data)
                                        client_id = model["client_id"]
                                        self.clients[client_id]["model"] = model["model"]
                                        self.clients[client_id]["model_received"] = True
                                        client_port = self.clients[client_id]["port"]
                                        self.send_confirmation(client_port)
                                        print(f"Getting local model from client {client_id.strip('client')}")
                                    except Exception as e:
                                        print(f"Failed: {e}")
                                        break
                        except Exception as e:
                            print(f"Error listening to client: {e}")
                            break
                    server_socket.close()
            except Exception as e:
                print(f"Can't connect to listener socket: {e}")
                break 
    
    def send_model(self) -> None:
        """
        Send global model to all connected clients in the current iteration.
        """
        # define message
        message = {
            "model": self.model,
            "iteration": self.iteration,
        }
        # send messages
        for client in self.clients:
            # only broadcast to sub clients for this round ONLY IF subsampling is on
            if len(self.subsamp_clients) > 0 and client not in self.subsamp_clients:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    #print(f"Sending global model to {client}")
                    server_socket.connect((HOST, self.clients[client]["port"]))
                    # send binary 0 to inform client to expect a model
                    server_socket.sendall(b"0")
                    server_socket.sendall(pickle.dumps(message))
                    server_socket.close()
            except Exception as e:
                print(f"Error sending to client: {e}")
                exit()
    
    def send_confirmation(self, port: str) -> None:
        """
        Send a confirmation message to a client to confirm a model has been received.

        Args:
            port (str): port of the client to be contacted.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.connect((HOST, port))
                # binary 1 represents a confirmation message
                server_socket.sendall(b"1")
                server_socket.close()
        except Exception as e:
            print(f"Error sending confirmation to client: {e}")
            exit()
    
    def federate(self) -> None:
        """
        Perform all the steps of the federated learning algorithm.
        """
        print(f"Waiting for {self.wait} seconds for all clients to join")
        time.sleep(self.wait)  # wait for clients to join
        for t in range(self.T):
            self.iteration = t
            self.clients = self.client_stack.copy()  # update client list
            print("Broadcasting new global model")
            if self.subsamp == 0:
                clients = self.clients
            else:
                clients = self.random_clients(self.subsamp)
            sender_thread = threading.Thread(target=self.send_model)
            sender_thread.start()
            print(f"\nGlobal Iteration {t + 1}:")
            print(f"Total Number of clients: {len(self.clients)}")
            while not self.check():
                time.sleep(0.25)
            self.update(clients)
                
    def update(self, clients) -> None:
        """
        Performs all steps to update the global model based on the local models provided
        and using the subsampling defined by self.subsamp where 0 means no subsampling.
        """
        print("Aggregating new global model")
        self.subsampled_update(clients)

        #if self.subsamp == 0:
            #self.subsampled_update(self.clients)
        #else:
            #clients = self.random_clients(self.subsamp)
            #self.subsampled_update(clients)

        # Reset model received flag
        for client in self.clients:
            self.clients[client]["model_received"] = False
    
    def subsampled_update(self, clients: dict) -> None:
        """
        Updates the global model using the local models from the client dictionary
        provided.

        Args:
            clients (dict): clients to be included in creating the new global model.
        """
        # calculate total data in all clients databases
        total_data = 0
        for client in clients:
            total_data += clients[client]["data_size"]
        # create empty weight and bias Tensors to be added to
        weight = None
        bias = None
        for client in clients:
            weight = torch.zeros_like(clients[client]["model"].weight, requires_grad=True)    
            bias = torch.zeros_like(clients[client]["model"].bias, requires_grad=True)
            break
        # calculate the new weight and bias by adding each client model averaged by the size of its database
        for client in clients:
            with torch.no_grad():
                weight += (clients[client]["data_size"] / total_data) * clients[client]["model"].weight 
                bias += (clients[client]["data_size"] / total_data) * clients[client]["model"].bias 
        # update the global model
        with torch.no_grad():
            self.model.weight.copy_(weight)
            self.model.bias.copy_(bias)
            
    def random_clients(self, size: int) -> dict:
        """
        Randomly selects clients and puts them into a dictionary for use in the subsampled model
        update method.

        Args:
            size (int): amount of clients needed by the subsampling

        Returns:
            dict: random dictionary of clients of the given size
        """
        clients = dict()
        fake = self.clients.copy()
        self.subsamp_clients.clear()

        # check if size provided is larger than available clients
        if size >= len(self.clients.keys()):
            print(f"Only {len(self.clients.keys())} are available. No subsampling is being performed.")
            return self.clients

        # add clients to dictionary
        while len(clients) < size:
            client = random.choice(list(fake.keys()))
            client_details = fake.pop(client)
            clients.update({client: client_details})
            self.subsamp_clients.append(client)

        #print("Randomly subsampled clients: ", self.subsamp_clients)
        return clients

    def check(self) -> bool:
        """
        Check that all clients included in the global iteration have returned
        a local model.

        Returns:
            bool: True if all client have returned a local model.
        """
        # No clients
        if len(self.clients) == 0:
            return False

        # Check only sub clients
        if len(self.subsamp_clients) > 0:
            for client in self.subsamp_clients:
                if not self.clients[client].get("model_received", False):
                    return False
        else:  # Check all clients
            for client in self.clients:
                if not self.clients[client].get("model_received", False):
                    return False

        return True
    
if __name__ == "__main__":
    port = int(sys.argv[1])
    subsamp = int(sys.argv[2])
    server = Server(port, subsamp)
    server.start()