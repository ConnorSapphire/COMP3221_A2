import sys
import socket
import threading

HOST = "localhost"
SERVER_PORT = 6000

class Client:
    def __init__(self, id, port, opt_method):
        self.client_id = id
        self.port = port
        self.opt_method = opt_method

    def connect(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, SERVER_PORT))
                data_rev = s.recv(1024)
                print(data_rev.decode('utf-8'))
                s.close()
        except:
            print("Can't connect to the Socket")


def main(id, port, opt_method):
    client = Client(id, port, opt_method)
    client.connect()


print("Client")
if __name__ == "__main__":
    id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    main(id, port, opt_method)