import sys
import socket
import threading
import json
import torch
import torch.nn as nn
import pickle

HOST = "localhost"

class Server:
    def __init__(self, port, subsamp):
        self.port = port
        self.subsamp = subsamp
        self.clients = {}
        self.listener_threads = []
        self.sender_threads = []
        self.stop_event = threading.Event()
        self.T = 10
        # self.w = torch.randn(8, 1,requires_grad=True)
        # self.b = torch.randn(1,requires_grad=True)
        self.model = nn.Linear(8, 1)

    def start(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((HOST, self.port))
                server_socket.listen()
                print(f"Server listening on port {self.port}")

                # Connect all clients
                while len(self.clients) < 5:
                    conn, addr = server_socket.accept()
                    print(f"Got connection from {addr}")
                    listener_thread = threading.Thread(target=self.listen_to_client, args=(conn, ))
                    self.listener_threads.append(listener_thread)
                    listener_thread.start()
                    sender_thread = threading.Thread(target=self.send)
                    self.sender_threads.append(sender_thread)
                    sender_thread.start()
                    

                print("Clients are connected")
        except Exception as e:
            print(f"Can't connect to listener socket: {e}")

    def stop(self):
        try:
            self.stop_event.set()
            for thread in self.listener_threads:
                thread.join()
        except:
            pass

    def listen_to_client(self, conn):  # Listen on port 6000
        with conn:
            while not self.stop_event.is_set():
                try:
                    message = conn.recv(1024)
                    if not message:
                        # TODO: This doesn't mean a client disconnected???
                        # print(f"Client disconnected")
                        break
                    message = json.loads(message.decode("utf-8"))
                    client_id = message["client_id"]
                    client_data_size = message["data_size"]
                    client_port = message["port"]
                    content = message["content"]

                    if message['content'] == "CONNECTION ESTABLISHED":
                        print(f"== Handshake: handle {client_id} connection ==")
                        self.clients[client_id] = {"port": client_port, "data_size": client_data_size}
                        print(self.clients)
                    else:
                        print(f"Message from {client_id} on port {client_port}: {content}")
                except:
                    print(f"Error listening to client")
                    break
    
    def send(self):
        message = {
            "model": self.model
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                while not self.stop_event.is_set():
                    for client in self.clients:
                        server_socket.connect((HOST, self.clients[client]["port"]))
                        server_socket.sendall(pickle.dumps(message))
                        print("Message sent")
        except Exception as e:
            print(f"Error sending to client: {e}")
            exit()
                
    def update(self):
        if self.subsamp == 0:
            self.subsampled_update(self.clients)
        else:
            clients = self.random_clients(self.subsamp)
            self.subsampled_update(self.clients)
    
    def subsampled_update(self, clients):
        total_data = 0
        for client in clients:
            total_data += client["data_size"]
        for client in clients:
            # += (client["data_size"] / total_data) * client["model"] 
            pass
            
    def random_clients(self, size):
        pass

if __name__ == "__main__":
    port = int(sys.argv[1])
    subsamp = int(sys.argv[2])
    server = Server(port, subsamp)
    server.start()