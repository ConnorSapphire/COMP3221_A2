import sys

import COMP3221_Messages as messages

class Server:
    def server():
        pass

def main():
    # Check program arguments
    if len(sys.argv) < 3:
        print(messages.make_error(messages.SERVER_NOT_ENOUGH_ARGS))
        return
    if not sys.argv[1].isnumeric() or not sys.argv[2].isnumeric():
        print(messages.make_error(messages.SERVER_NOT_NUMERIC))
        return
    port = int(sys.argv[1])
    if port < 1024 or port > 65535:
        print(messages.make_error(messages.SERVER_PORT_BOUNDS))
        return
    subsamp = int(sys.argv[2])
    # TODO: check bounds for subsamp.


if __name__ == "__main__":
    main()