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
        self.client_stack = {}
        self.listener_threads = []
        self.stop_event = threading.Event()
        self.T = 100
        self.model = nn.Linear(8, 1)
        self.wait = 5

    def start(self):
        federated_thread = threading.Thread(target=self.federate, daemon=True)
        federated_thread.start()
        listener_thread = threading.Thread(target=self.listen_to_client)
        self.listener_threads.append(listener_thread)
        listener_thread.start()

    def stop(self):
        try:
            self.stop_event.set()
            for thread in self.listener_threads:
                thread.join()
        except:
            pass

    def listen_to_client(self):  # Listen on port 6000
        print(f"Server listening on port {self.port}")
        while not self.stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.bind((HOST, self.port))
                    server_socket.listen(5)
                
                    # Connect all clients
                    conn, addr = server_socket.accept()
                    with conn:
                        try:
                            message = conn.recv(1)
                            if message == b"0":
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
                                    print(f"== Handshake: handle {client_id} connection ==")
                                    if len(self.client_stack) < 5:
                                        self.client_stack[client_id] = {"port": client_port, "data_size": client_data_size}
                                    else:
                                        print("== Client ignored : Too many clients ==")
                                else:
                                    print(f"\tMessage from {client_id} on port {client_port}: {content}")
                            elif message == b"1":
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
                                        self.send_confirmation(client_id)
                                        print(f"\tServer received local model data from {client_id}")
                                    except Exception as e:
                                        print(f"Failed: {e}")
                                        break
                        except Exception as e:
                            print(f"Error listening to client: {e}")
                            break
                    # print(f"Got connection from {addr}")
                    server_socket.close()
            except Exception as e:
                print(f"Can't connect to listener socket: {e}")
                break 
    
    def send_model(self) -> None:
        message = {
            "model": self.model
        }
        for client in self.clients:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    print(f"\tSending model to {client} at {self.clients[client]["port"]}")
                    server_socket.connect((HOST, self.clients[client]["port"]))
                    server_socket.sendall(b"0")
                    server_socket.sendall(pickle.dumps(message))
                    server_socket.close()
            except Exception as e:
                print(f"Error sending to client: {e}")
                exit()
    
    def send_confirmation(self, client: str) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.connect((HOST, self.clients[client]["port"]))
                server_socket.sendall(b"1")
                server_socket.close()
        except Exception as e:
            print(f"Error sending confirmation to client: {e}")
            exit()
    
    def federate(self) -> None:
        print(f"Waiting for {self.wait} seconds for all clients to join")
        time.sleep(self.wait)
        for t in range(self.T):
            self.clients = self.client_stack.copy()
            print(f"Global iteration {t + 1}:")
            sender_thread = threading.Thread(target=self.send_model)
            sender_thread.start()
            while not self.check():
                time.sleep(0.25)
            self.update()        
                
    def update(self) -> None:
        print("\tUpdating global model")
        if self.subsamp == 0:
            self.subsampled_update(self.clients)
        else:
            clients = self.random_clients(self.subsamp)
            self.subsampled_update(clients)
        for client in self.clients:
            self.clients[client].pop("model")
    
    def subsampled_update(self, clients: dict) -> None:
        print("\tCalculating new global model")
        total_data = 0
        for client in clients:
            total_data += clients[client]["data_size"]
        
        weight = None
        bias = None
        for client in clients:
            weight = torch.zeros_like(clients[client]["model"].weight, requires_grad=True)    
            bias = torch.zeros_like(clients[client]["model"].bias, requires_grad=True)
            break
        
        for client in clients:
            with torch.no_grad():
                weight += (clients[client]["data_size"] / total_data) * clients[client]["model"].weight 
                bias += (clients[client]["data_size"] / total_data) * clients[client]["model"].bias 
                
        with torch.no_grad():
            self.model.weight.copy_(weight)
            self.model.bias.copy_(bias)
        print("\tGlobal model updated")
            
    def random_clients(self, size: int) -> dict:
        print(f"\tRandomly subsampling {size} client models")
        clients = dict()
        fake = self.clients.copy()
        if size >= len(self.clients.keys()):
            print(f"Only {len(self.clients.keys())} are available. No subsampling is being performed.")
            self.subsampled_update(self.clients)
        while len(clients) < size:
            client = random.choice(list(fake.keys()))
            client_details = fake.pop(client)
            clients.update({client: client_details})
        return clients

    def check(self) -> bool:
        if len(self.clients) == 0:
            return False
        for client in self.clients:
            if not "model" in self.clients[client]:
                return False
        return True
    
if __name__ == "__main__":
    port = int(sys.argv[1])
    subsamp = int(sys.argv[2])
    server = Server(port, subsamp)
    server.start()