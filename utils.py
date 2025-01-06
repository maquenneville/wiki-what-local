import configparser
import threading
import sys
import time
import os


def load_config():

    # Check if config file exists
    if not os.path.exists("config.ini"):
        raise FileNotFoundError("The config file 'config.ini' was not found.")

    class Args:
        """Class to hold configuration arguments."""

        pass

    config = configparser.ConfigParser()
    config.read("config.ini")

    args = Args()

    for section in config.sections():
        for key, value in config.items(section):
            # Check for boolean strings and convert them
            if value.lower() in ["true", "false"]:
                setattr(args, key.lower(), value.lower() == "true")
            else:
                setattr(args, key.lower(), value)

    return args


class Spinner:
    def __init__(self, message="Thinking..."):
        self._message = message
        self._running = False
        self._spinner_thread = None

    def start(self):
        self._running = True
        self._spinner_thread = threading.Thread(target=self._spin)
        self._spinner_thread.start()

    def stop(self):
        self._running = False
        self._spinner_thread.join()

    def _spin(self):
        spinner_chars = "|/-\\"
        index = 0

        while self._running:
            sys.stdout.write(
                f"\r{self._message} {spinner_chars[index % len(spinner_chars)]}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            index += 1

        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._message) + 2))
        sys.stdout.flush()
