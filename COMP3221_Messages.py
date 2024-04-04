from colorama import Fore, Style

SERVER_NOT_ENOUGH_ARGS = """Error: not enough arguments provided.
Please call the program using:
$python COMP3221_FLServer.py <PORT-SERVER> <SUB-CLIENT>
Where   PORT-SERVER is the port the server is running on.
        SUB-CLIENT is the amount of subsampling between 0 and K (exclusive). Input 0 to disable subsampling."""
SERVER_NOT_NUMERIC = """Error: <PORT-SERVER> and <SUB-CLIENT> must be numeric."""
SERVER_PORT_BOUNDS = """Error: Port must be between 1024 and 65535."""

def make_error(string: str) -> str:
    return f"{Fore.MAGENTA}{string}{Style.RESET_ALL}"

