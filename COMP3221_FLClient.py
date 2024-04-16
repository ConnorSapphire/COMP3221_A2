import sys
import socket
import threading
import json

HOST = "localhost"
SERVER_PORT = 6000

class Client:
    def __init__(self, id, port, opt_method):
        self.client_id = id
        self.port = port
        self.opt_method = opt_method
        self.stop_event = threading.Event()

    def start(self):
        self.listener_thread = threading.Thread(target=self.listen_to_server)
        self.listener_thread.start()
        self.send(f"CONNECTION ESTABLISHED")

    def stop(self):
        try:
            self.stop_event.set()
            self.listener_thread.join()
        except:
            pass

    def listen_to_server(self):  # Listen on port 6001, 6002, etc.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.bind((HOST, self.port))
                client_socket.listen()
                print(f"Client listening on port {self.port}")

                conn, addr = client_socket.accept()
                with conn:
                    data = conn.recv(1024)
                    if data:
                        print("Client received data")
        except:
            print("Can't connect to the listener socket")

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
        except:
            print("Message failed")


id = sys.argv[1]
port = int(sys.argv[2])
opt_method = int(sys.argv[3])
client = Client(id, port, opt_method)
client.start()