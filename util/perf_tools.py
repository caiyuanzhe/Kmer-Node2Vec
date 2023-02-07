import arrow
import sys
import os
import psutil
import time

default_log = f"log/{arrow.utcnow().format('YYYYMMDD-HHmm')}-KMerGraph2Vec-Info.log"


class Tee(object):
    """ Logger for parameters and execution info of function """

    def __init__(self, fname=default_log):
        self.terminal = sys.stdout
        self.log = open(fname, 'w')

    def write(self, message):  # for Timer wrapper
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Timer:
    """Timer for logging runtime of function."""

    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __call__(self, func):
        """Call timer decorator."""

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            hrs = int(duration // 3600)
            mins = int(duration % 3600 // 60)
            secs = duration % 60
            print(f"Took {hrs:02d}:{mins:02d}:{secs:05.2f} to {self.name}")

            return result

        return wrapper if self.verbose else func


def mem_info():
    p = psutil.Process(os.getpid())
    info = p.memory_full_info()
    memory = info.uss  # bytes
    return memory



