import zmq 
from .utils import * 
import numpy as np 
import pytest
from  multiprocessing import Process
import time

def test_send_and_receive_data():
    def send_data():
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.connect("tcp://127.0.0.1:5555")
        print('sender bound')
        A = np.array([[[1,1,1,]]])
        sentence = 'guineapig sentence'
        send_array_and_str(sender, A, sentence)
        print('msg sent')

    def receive_data():
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://127.0.0.1:5555")
        print('receiver bound')
        while True:
            sentence, A = recv_array_and_str(receiver)
            print(sentence, A)
            assert sentence == 'guineapig sentence'
            assert A.all() == np.array([[[1,1,1,]]]).all()
    
    receive = Process(target=receive_data)
    receive.start()
    send = Process(target=send_data)
    send.start()
    time.sleep(1)
    receive.terminate()
    send.terminate()

class TestSinkJob:
    def __init__(self):
        from mbert_sink import SinkJob
        self.sink_job = SinkJob()
    def test_add_embed(self):
        progress_embeds = 0
        progress = 0
        checksum = 0
        pid = 0 
        _pending_embeds = []
        data = np.random.rand(1,4,768)
        final_ndarray = np.zeros(data.shape)
        pid = 1
        sink_job.add_embed(data, pid)
        assert final_ndarray.shape == (1,4,768)
        assert progress_embeds == 1

def fill():
    progress_embeds = 0
    progress = 0
    checksum = 0
    pid = 0 
    _pending_embeds = []
    data = np.random.rand(1,4,768)
    final_ndarray = np.zeros(data.shape)
    pid = 0
    sink_job = SinkJob()
    sink_job.add_embed(data, pid)
    assert final_ndarray.all() == data.all()
    assert progress_embeds == 1













    
        

