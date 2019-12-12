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
        client = context.socket(zmq.PULL).bind("tcp://*:5555")
        # bind ventilator to worker to allocate jobs
        sender = context.socket(zmq.PUSH).bind("tcp://*:5557")
        # bind ventilator to sink
        sink = context.socket(zmq.PUSH).bind("tcp://localhost:5558")

        # give everything a second to spin up and connect
        time.sleep(1)
        print('ventilator is starting')
        while True:
            sentence, tokens_tensor = recv_array_and_str(client)
            print("received request")
            send_array_and_str(sender, tokens_tensor, sentence)
            print("sent request")
            time.sleep(1)