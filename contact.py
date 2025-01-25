import socket
import time

HOST = '127.0.0.1'
PORT = 5001

# Construct the command
command = "place 26 26 21 0 2499 10500"
class SendCommand():

    def __init__(self, cur_id, host=HOST, port=PORT):
        self.id = cur_id
        self.host = host
        self.port = port
        self.sock = None
        self.class_dict = {
            0: 27,
            1: 28,
            2: 26,
            3: 26,
            4: 26,
            5: 28,
            6: 26,
            7: 26
        }
        self.instance_dict = {
            0: 0,
            1: 0,
            2: 21,
            3: 38,
            4: 30,
            5: 11,
            6: 14,
            7: 10
        }
        self.spellindex_dict = {
            2:0,
            7:1,
            3:2,
            6:3,
            4:4,
            0:5,
            5:6,
            1:7

        }
        self.tick = 0
    
    def connect(self):
        """
        Connect to the server if not already connected,
        or if an existing connection has been lost.
        """
        # If not connected, or was closed, create a new socket and connect.
        if not self.is_connected():
            self.close()  # Ensures any half-open socket is closed
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))

    def is_connected(self):
        """
        Check if the socket is currently connected.
        This will attempt to send an empty byte-string. If an error occurs,
        it means the connection is broken. If sock is None, it's not connected.
        """
        if not self.sock:
            return False

        try:
            # The remote end might see this as a "do-nothing" send;
            # if the connection is broken, this should raise an exception.
            self.sock.sendall(b'')
            return True
        except socket.error:
            return False
        
    def place_troop(self, class_id, x, y):
        command = f"place {self.id} {self.class_dict[class_id]} {self.instance_dict[class_id]} {self.spellindex_dict[class_id]} {x} {y}"
        if not self.is_connected():
            raise ConnectionError("Not connected. Call connect() first.")

        self.sock.sendall(command.encode('utf-8'))
        print('sending command')
        return self.read_response()

    def get_tick(self):
        self.tick = self.read_response()
        return self.tick
    
    def read_response(self):
        """
        Receive up to 1024 bytes from the server and return decoded text.
        Raises ConnectionError if not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected. Call connect() first.")

        data = self.sock.recv(1024)
        return data.decode('utf-8')

    def close(self):
        """
        Close the socket if it’s open.
        """
        if self.sock:
            try:
                self.sock.close()
            except socket.error:
                pass
            self.sock = None
    
    
if __name__ == "__main__":
    s = SendCommand(26)
    s.connect()

    s.place_troop(1, 5000, 5000)

    