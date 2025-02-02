import socket
import time

HOST = '127.0.0.1'
PORT = 5001

class SendCommand:
    def __init__(self, cur_id, host=HOST, port=PORT):
        self.id = cur_id
        self.host = host
        self.port = port
        self.sock = None
        
        # Troop info dicts
        self.class_dict = {
            0: 27, 1: 28, 2: 26, 3: 26,
            4: 26, 5: 28, 6: 26, 7: 26
        }
        self.instance_dict = {
            0: 0, 1: 0, 2: 21, 3: 38,
            4: 30, 5: 11, 6: 14, 7: 10
        }
        self.spellindex_dict = {
            2: 0, 7: 1, 3: 2, 6: 3,
            4: 4, 0: 5, 5: 6, 1: 7
        }

        # Internal state
        self.inbattle = False
        self.isBattleMaster = None
        self.tick = None
        self.max_retries = 5
        
        # Track the last troop class that was placed
        self.last_placed_class_id = None

    def connect(self):
        """
        Connect to the server if not already connected.
        Perform the "hello <id>" handshake.
        """
        if not self.is_connected():
            self.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            
            # Send handshake
            handshake_cmd = f"hello {self.id}\n"
            self.sock.sendall(handshake_cmd.encode('utf-8'))
            
            # Parse response
            resp = self._read_response(timeout=2.0)
            if resp is None:
                raise ConnectionError("No response from server during handshake.")
            if resp.startswith("ERROR:"):
                raise ConnectionError(f"Server error during handshake: {resp}")
            
            print("Handshake response:", resp.strip())

    def is_connected(self):
        """
        Returns True if we believe the socket is connected (no error on sendall).
        """
        if not self.sock:
            return False

        try:
            self.sock.sendall(b'')  # "no-op" send
            return True
        except socket.error:
            return False

    def place_troop(self, class_id, x, y):
        """
        Send a 'place' command to position a troop for this player.
        Before sending, we check if this is the same class_id as the last one
        to avoid accidentally sending duplicate 'place' commands.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected. Call connect() first.")
        
        # Check if we're about to send the same place command as last time
        if self.last_placed_class_id == class_id:
            print(
                f"Skipping place command because the last placed card "
                f"had the same class_id={class_id}."
            )
            return
        
        cmd = (
            f"place {self.id} {self.class_dict[class_id]} "
            f"{self.instance_dict[class_id]} {self.spellindex_dict[class_id]} "
            f"{x} {y}\n"
        )
        self.sock.sendall(cmd.encode('utf-8'))
        print("Sending place command:", cmd.strip())
        
        # Update the last placed card ID to avoid duplication
        self.last_placed_class_id = class_id

        resp = self._read_response(timeout=2.0)
        self._handle_generic_response(resp, "Placing troop")

    def start_battle(self):
        """
        Send 'battle <playerId>' to force-start a match.
        Raises an exception if there's an error.
        
        Note: This method does NOT automatically update self.inbattle.
              Instead, call check_inbattle() afterwards to refresh inbattle/battleMaster status.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected. Call connect() first.")

        cmd = f"battle {self.id}\n"
        self.sock.sendall(cmd.encode('utf-8'))
        print("Sending battle command:", cmd.strip())

        resp = self._read_response(timeout=2.0)
        self._handle_generic_response(resp, "Starting battle")

    def check_inbattle(self):
        """
        Send 'inbattle <playerId>' to see if we're in a battle + BattleMaster status.
        Updates self.inbattle and self.isBattleMaster accordingly.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected. Call connect() first.")

        cmd = f"inbattle {self.id}\n"
        self.sock.sendall(cmd.encode('utf-8'))
        print("Sending inbattle command:", cmd.strip())

        resp = self._read_response(timeout=2.0)
        if resp is None:
            raise ConnectionError("No response from server for inbattle check.")

        if resp.startswith("ERROR:"):
            raise ConnectionError(f"Server error checking inbattle: {resp}")

        # Expected format: "OK: isInBattle=true, isBattleMaster=false"
        line = resp.strip()
        if not line.startswith("OK:"):
            raise ValueError(f"Unexpected inbattle response (no 'OK:'): {line}")

        inbattle_str = self._extract_value(line, "isInBattle=")
        battlemaster_str = self._extract_value(line, "isBattleMaster=")

        if not inbattle_str:
            raise ValueError("Server did not return 'isInBattle=' in the expected format.")
        if not battlemaster_str:
            raise ValueError("Server did not return 'isBattleMaster=' in the expected format.")

        self.inbattle = (inbattle_str.lower() == "true")
        if not self.inbattle:
            self.isBattleMaster = None
            self.tick = None
        else:
            self.isBattleMaster = (battlemaster_str.lower() == "true")

    def request_tick(self):
        """
        Sends 'tick <playerId>' to request the current battle time from the server.
        If the server says '0' (not in a battle or not found), self.tick = None if inbattle == False.
        Otherwise, parses the integer tick.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected. Call connect() first.")

        cmd = f"tick {self.id}\n"
        for attempt in range(self.max_retries):
            try:
                print(f"Requesting tick (attempt {attempt+1}/{self.max_retries}): {cmd.strip()}")
                self.sock.sendall(cmd.encode('utf-8'))

                resp = self._read_response(timeout=2.0)
                if resp is None:
                    print("Warning: No response from server for tick request.")
                    continue  # retry

                if resp.startswith("ERROR:"):
                    print(f"Warning: Server error while requesting tick: {resp}")
                    break  # often not transient, so break

                resp_stripped = resp.strip()
                try:
                    tick_value = int(resp_stripped)
                except ValueError:
                    print(f"Warning: Unexpected tick format: '{resp_stripped}'")
                    break

                # Successfully parsed a tick value
                if tick_value == 0 and not self.inbattle:
                    self.tick = None
                else:
                    self.tick = tick_value

                # Request was successful; exit the retry loop
                return

            except Exception as e:
                print(f"Warning: Error during tick request: {e}")
                time.sleep(0.2)

    def close(self):
        """Close the socket if open."""
        if self.sock:
            try:
                self.sock.close()
            except socket.error:
                pass
            self.sock = None

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _read_response(self, timeout=5.0):
        """
        Internal helper to read up to 1024 bytes with a specified timeout.
        Returns the decoded string or None if timed out or connection closed.
        """
        if not self.is_connected():
            raise ConnectionError("Socket is not connected.")

        self.sock.settimeout(timeout)
        try:
            data = self.sock.recv(1024)
            if not data:
                # Server closed the connection
                return None
            return data.decode('utf-8')
        except socket.timeout:
            return None

    def _handle_generic_response(self, resp, action_label):
        """
        Handles typical command responses.
        If resp is None => raise exception.
        If "ERROR:" => raise.
        Otherwise, print success.
        """
        if resp is None:
            raise ConnectionError(f"No response from server while {action_label.lower()}.")
        if resp.startswith("ERROR:"):
            raise ConnectionError(f"Server error {action_label.lower()}: {resp.strip()}")
        
        print(f"{action_label} succeeded: {resp.strip()}")

    def _extract_value(self, line, key):
        """
        Naive string extraction for something like: "... isInBattle=true..."
        and you pass key="isInBattle=". Returns the substring up to a comma or end-of-line.
        """
        start_idx = line.find(key)
        if start_idx == -1:
            return ""

        # Skip to end of key
        start_idx += len(key)

        # Find next comma or end of line
        comma_idx = line.find(',', start_idx)
        if comma_idx == -1:
            # No comma found; take the rest of the line
            return line[start_idx:].strip()
        else:
            # Slice up to the comma
            return line[start_idx:comma_idx].strip()


def main():
    s = SendCommand(29)  # Example: track player with ID 29

    try:
        # 1) Connect & handshake
        s.connect()

        # 2) Check in-battle status
        s.check_inbattle()
        print(f"[Initial] inbattle={s.inbattle}, battleMaster={s.isBattleMaster}")

        # 3) Attempt to request the current tick right away (likely 0 if no battle)
        s.request_tick()
        print(f"Tick (initial): {s.tick}")

        # 4) Force a battle
        s.start_battle()

        # 5) Check inbattle status again
        s.check_inbattle()
        print(f"[After forcing battle] inbattle={s.inbattle}, battleMaster={s.isBattleMaster}")

        # 6) Place some troops.
        s.place_troop(0, 5, 5)  # first time
        s.place_troop(0, 5, 6)  # this should skip if we don't allow repeated class_id

        # 7) Request tick in a loop (example)
        while True:
            s.request_tick()
            print(f"Tick: {s.tick}")
            time.sleep(1)

    except ConnectionError as ex:
        print("A connection error occurred:", ex)
    finally:
        s.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
