# send and receive numpy arrays on ZeroMQ 
import zmq
import numpy
import torch
import json
import logging
import argparse 
import os
import uuid

def send_array_and_str(socket, A, sentence, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype = str(A.dtype),shape = A.shape)
    socket.send_string(sentence, flags|zmq.SNDMORE)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array_and_str(socket, flags=0, copy=True, track=False):
    """recv a numpy array and sentence"""
    sentence = socket.recv_string(flags=flags)
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return sentence, A.reshape(md['shape'])

def preprocess(text, tokenizer):
    '''tokenize text into subwords and convert to indices
    :param text str: text to be preprocessed 
    :param tokenizer: BertTokenizer object
    :output: torch tensor of vocab ids
    '''
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).numpy()
    return tokens_tensor

def auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://127.0.0.1')
    else:
        # Get the location for tmp file for sockets
        try:
            tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
            if not os.path.exists(tmp_dir):
                raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
            tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
        except KeyError:
            tmp_dir = '*'

        socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')

class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)

def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger

def check_max_seq_len(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        ivalue = int(value)
        if ivalue <= 3:
            raise argparse.ArgumentTypeError("%s is an invalid int value must be >3 "
                                             "(account for maximum three special symbols in BERT model) or NONE" % value)
    except TypeError:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue

def get_args_parser():
    parser = argparse.ArgumentParser(description='start a BertServer for serving')
    parser.add_argument('-max_seq_len', type=check_max_seq_len, default=25,
                        help='maximum length of a sequence, longer sequence will be trimmed on the right side. '
                             'set it to NONE for dynamically using the longest sequence in a (mini)batch.')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    parser.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    parser.add_argument('-max_batch_size', type=int, default=256,
                        help='maximum number of sequences handled by each worker')
    parser.add_argument('-priority_batch_size', type=int, default=16,
                        help='batch smaller than this size will be labeled as high priority,'
                             'and jumps forward in the job queue')
    parser.add_argument('-cpu', action='store_true', default=False,
                        help='running on CPU (default on GPU)')
    parser.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determine the fraction of the overall amount of memory \
                        that each visible GPU should be allocated per worker. \
                        Should be in range [0.0, 1.0]')
    parser.add_argument('-device_map', type=int, nargs='+', default=[],
                        help='specify the list of GPU device ids that will be used (id starts from 0). \
                        If num_worker > len(device_map), then device will be reused; \
                        if num_worker < len(device_map), then device_map[:num_worker] will be used')
    parser.add_argument('-prefetch_size', type=int, default=10,
                        help='the number of batches to prefetch on each worker. When running on a CPU-only machine, \
                        this is set to 0 for comparability')
    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    return parser