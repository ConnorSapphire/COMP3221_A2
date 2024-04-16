import sys
import socket
import threading
import json

import COMP3221_Messages as messages

HOST = "localhost"

class Server:
    def __init__(self, port, subsamp):
        self.port = port
        self.subsamp = subsamp
        self.clients = {}
        self.listener_threads = []
        self.stop_event = threading.Event()

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

                print("Clients are connected")
        except:
            print("Can't connect to listener socket")

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
                        print(f"Client disconnected")
                        break
                    message = json.loads(message.decode("utf-8"))
                    client_id = message["client_id"]
                    client_port = message["port"]
                    content = message["content"]

                    if message['content'] == "CONNECTION ESTABLISHED":
                        print(f"== Handshake: handle {client_id} connection ==")
                        self.clients[client_id] = {
                            "port": client_port
                        }
                        print(self.clients)
                    else:
                        print(f"Message from {client_id} on port {client_port}: {content}")
                except:
                    print(f"Error listening to client")
                    break


port = int(sys.argv[1])
subsamp = int(sys.argv[2])
server = Server(port, subsamp)
server.start()