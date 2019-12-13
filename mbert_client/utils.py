# send and receive numpy arrays on ZeroMQ 
import zmq
import numpy
import torch
import json

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