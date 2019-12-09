import multiprocessing
import os 
import random
import sys
import threading 
import time 
from collections import defaultdict
from datetime import datetime
from itertools import chain 
from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np 
import zmq 
from zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi 

__all__ = ['__version__', 'MBertServer']
__version__ = '0.1'

class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    show_status = b'SHOW_STATUS'
    new_job = b'REGISTER'
    data_token = b'TOKENS'
    data_embed = b'EMBEDDINGS'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


class MBertServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }

class MBertWorker():

class MBertSink():
