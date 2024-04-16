import sys
import socket
import threading

import COMP3221_Messages as messages

HOST = "localhost"

class Server:
    def __init__(self, port, subsamp):
        self.port = port
        self.subsamp = subsamp
        self.clients = {}

    def connect(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((HOST, self.port))
                server_socket.listen()
                print(f"Server listening on port {self.port}")

                # Connect all clients
                while len(self.clients) < 5:
                    conn, addr = server_socket.accept()
                    print(f"Got connection from {addr}")
                    client_handler_thread = threading.Thread(target=self.handle_client, args=(conn, ))
                    client_handler_thread.start()
                    self.clients[addr] = conn
                print("All clients connected")
        except:
            print("Server error connecting to socket")

    def handle_client(self, conn):
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("No data received")
                    break
                print("Got data")


def main(port, subsamp):
    server = Server(port, subsamp)
    server.connect()


print("Server")
if __name__ == "__main__":
    port = int(sys.argv[1])
    subsamp = int(sys.argv[2])
    main(port, subsamp)