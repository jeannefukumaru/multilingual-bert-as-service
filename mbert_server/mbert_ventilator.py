import zmq 
import random 
import time 
import numpy as np 
import time 
import torch 
from utils import send_array_and_str, recv_array_and_str, preprocess
from argparse import ArgumentParser 

def run_ventilator(): 
    # zmq setup 
    context = zmq.Context()
    # bind ventilator to client to accept sentence strings 
    client = context.socket(zmq.PULL)
    client.bind('tcp://127.0.0.1:5555')
    print('client bound')
    # bind ventilator to worker to allocate jobs
    sender = context.socket(zmq.PUSH)
    sender.bind('tcp://127.0.0.1:5557')
    print('sender bound')
    # bind ventilator to sink
    sink = context.socket(zmq.PUSH).bind('tcp://127.0.0.1:5558')
    print('sink bound')
    # give everything a second to spin up and bind
    time.sleep(5)
    print('ventilator is starting')
    print('waiting for incoming messages...')
    while True:
        sentence, tokens_tensor = recv_array_and_str(client)
        print("received request")
        print(sentence, tokens_tensor)
        send_array_and_str(sender, tokens_tensor, sentence)
        print("sent request")
        time.sleep(1)

if __name__=='__main__':
    run_ventilator()
