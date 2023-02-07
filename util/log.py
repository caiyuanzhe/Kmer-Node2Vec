import arrow
import sys

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

