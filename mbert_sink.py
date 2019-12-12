import sys
import time 
import zmq
from utils import recv_array_and_str

def run_sink():
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:5558")
    result = recv_array_and_str(receiver)
    print(f'result received:{result}')



